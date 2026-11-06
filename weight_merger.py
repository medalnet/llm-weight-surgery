"""
Weight Merger для GGUF моделей
Декодирует Q4_0 веса в float32, ищет пары между соседними слоями,
пробует слить два веса в один с минимальной ошибкой.
"""

import numpy as np
import re
import sys
from pathlib import Path

try:
    from gguf import GGUFReader, GGMLQuantizationType
    from gguf.quants import dequantize
except ImportError:
    print("Установи: pip install gguf")
    sys.exit(1)


def load_tensors_f32(path: str) -> dict:
    """Читаем все тензоры и декодируем в float32"""
    reader = GGUFReader(path)
    tensors = {}
    print(f"Загружаем и декодируем тензоры...")
    for t in reader.tensors:
        try:
            raw = t.data  # numpy array сырых байт/значений
            qtype = t.tensor_type
            # Декодируем из квантизации в float32
            f32 = dequantize(raw, qtype).astype(np.float32)
            # Reshape в правильную форму
            f32 = f32.reshape(t.shape)
            tensors[t.name] = {"data": f32, "shape": f32.shape}
        except Exception as e:
            # Некоторые тензоры могут быть уже float (norm weights и т.д.)
            try:
                f32 = raw.astype(np.float32).reshape(t.shape)
                tensors[t.name] = {"data": f32, "shape": f32.shape}
            except:
                pass  # пропускаем проблемные
    print(f"Загружено тензоров: {len(tensors)}")
    return tensors


def find_pairs(tensors: dict) -> list:
    """Ищем пары весов с одинаковым типом в соседних слоях (N и N+1)"""
    layer_map = {}
    for name in tensors:
        m = re.match(r"blk\.(\d+)\.(.*)", name)
        if m:
            idx = int(m.group(1))
            ttype = m.group(2)
            layer_map.setdefault(idx, {})[ttype] = name

    pairs = []
    indices = sorted(layer_map.keys())
    for i in range(len(indices) - 1):
        l1, l2 = indices[i], indices[i + 1]
        for ttype in layer_map.get(l1, {}):
            if ttype in layer_map.get(l2, {}):
                pairs.append({
                    "name_a": layer_map[l1][ttype],
                    "name_b": layer_map[l2][ttype],
                    "label": f"blk.{l1}-{l2}.{ttype}",
                })
    return pairs


def try_merge(a: np.ndarray, b: np.ndarray):
    """
    Пробуем слить два веса. Возвращает (merged, method, error).
    Метрика ошибки: MSE выхода на случайных входах.
    """
    best = (None, None, float('inf'))

    # --- Метод 1: среднее (только если формы одинаковые) ---
    if a.shape == b.shape:
        merged = (a + b) / 2.0
        err = shape_error(a, b, merged)
        if err < best[2]:
            best = (merged, "mean", err)

    # --- Метод 2: матричное произведение (перепрыгиваем нейрон) ---
    # a: [out_a, in_a], b: [out_b, in_b]
    # чтобы перепрыгнуть: нужно a.shape[0] == b.shape[1]
    if a.ndim == 2 and b.ndim == 2:
        if a.shape[0] == b.shape[1]:
            # b @ a: [out_b, in_a]
            try:
                merged = b @ a
                err = matmul_error(a, b, merged)
                if err < best[2]:
                    best = (merged, "matmul(b@a)", err)
            except:
                pass
        if a.shape[1] == b.shape[0]:
            # a @ b нет смысла для "перепрыгивания", но пробуем
            try:
                merged = a @ b  # [out_a, in_b]  — другая форма
                err = matmul_error_ab(a, b, merged)
                if err < best[2]:
                    best = (merged, "matmul(a@b)", err)
            except:
                pass

    # --- Метод 3: взвешенное по норме ---
    if a.shape == b.shape:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na + nb > 0:
            merged = (a * na + b * nb) / (na + nb)
            err = shape_error(a, b, merged)
            if err < best[2]:
                best = (merged, "weighted", err)

    return best


def shape_error(a, b, merged) -> float:
    """Ошибка для одинаковых форм: насколько merged близок к обоим"""
    err_a = np.mean((a - merged) ** 2)
    err_b = np.mean((b - merged) ** 2)
    return float((err_a + err_b) / 2)


def matmul_error(a, b, merged) -> float:
    """
    Ошибка для matmul: прогоняем вектор через a->b и через merged,
    сравниваем выходы.
    b@a имеет форму [out_b, in_a]
    x: [N, in_a] -> через a: [N, out_a=in_b] -> через b: [N, out_b]
    x: [N, in_a] -> через merged: [N, out_b]  (если merged = b@a)
    """
    try:
        in_size = a.shape[1]
        x = np.random.randn(200, in_size).astype(np.float32)
        # через два слоя: x @ a.T @ b.T
        out2 = (x @ a.T) @ b.T
        # через merged = b @ a: x @ merged.T
        out1 = x @ merged.T
        return float(np.mean((out2 - out1) ** 2))
    except:
        return float('inf')


def matmul_error_ab(a, b, merged) -> float:
    """Ошибка для a@b"""
    try:
        in_size = a.shape[1]
        x = np.random.randn(200, in_size).astype(np.float32)
        out2 = (x @ a.T) @ b.T
        out1 = x @ merged.T
        return float(np.mean((out2 - out1) ** 2))
    except:
        return float('inf')


def cosine_sim(a, b) -> float:
    if a.shape != b.shape:
        return 0.0
    af, bf = a.flatten(), b.flatten()
    n = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / n) if n > 0 else 0.0


def analyze(gguf_path: str, top_n: int = 20, limit_pairs: int = 100):
    print(f"\n{'='*60}")
    print(f"Файл: {gguf_path}")
    print(f"{'='*60}\n")

    tensors = load_tensors_f32(gguf_path)
    pairs = find_pairs(tensors)
    print(f"Пар весов между соседними слоями: {len(pairs)}")

    results = []
    total = min(limit_pairs, len(pairs))
    print(f"Анализируем {total} пар...\n")

    for i, p in enumerate(pairs[:total]):
        a = tensors[p["name_a"]]["data"]
        b = tensors[p["name_b"]]["data"]

        sim = cosine_sim(a, b)
        merged, method, error = try_merge(a, b)

        savings = 0.0
        if merged is not None:
            savings = (a.nbytes + b.nbytes - merged.nbytes) / 1024 / 1024

        results.append({
            "label":   p["label"],
            "shape_a": a.shape,
            "shape_b": b.shape,
            "sim":     sim,
            "method":  method or "—",
            "error":   error,
            "save_mb": savings,
        })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{total}...")

    # Сортируем: сначала те у кого есть метод и малая ошибка
    has_method = [r for r in results if r["method"] != "—" and r["error"] != float('inf')]
    no_method  = [r for r in results if r["method"] == "—" or r["error"] == float('inf')]
    has_method.sort(key=lambda x: x["error"])

    all_sorted = has_method + no_method

    print(f"\n{'='*70}")
    print(f"ТОП-{top_n} кандидатов:")
    print(f"{'='*70}")
    print(f"{'Пара':<38} {'Сходство':>8} {'Метод':>14} {'Ошибка MSE':>12} {'Экономия':>9}")
    print("-" * 85)

    for r in all_sorted[:top_n]:
        err = f"{r['error']:.6f}" if r['error'] != float('inf') else "    N/A"
        print(f"{r['label']:<38} {r['sim']:>8.4f} {r['method']:>14} {err:>12} {r['save_mb']:>7.2f} MB")

    if has_method:
        total_save = sum(r["save_mb"] for r in has_method)
        print(f"\nПар с рабочим слиянием: {len(has_method)}")
        print(f"Потенциальная экономия: {total_save:.1f} MB")

        best = has_method[0]
        print(f"\n✓ Лучший кандидат: {best['label']}")
        print(f"  Метод: {best['method']}, MSE: {best['error']:.8f}, сходство: {best['sim']:.4f}")
    else:
        print("\nНет пар с рабочим слиянием в этой выборке.")
        print("Скорее всего все веса разных форм — нужен другой подход.")

    return all_sorted


if __name__ == "__main__":
    path = r"F:\12\Sovenok-Hacker\Qwen3-0.6B-Q4_0-GGUF\qwen3-0.6b-q4_0.gguf"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    if not Path(path).exists():
        print(f"Файл не найден: {path}")
        sys.exit(1)
    analyze(path)
