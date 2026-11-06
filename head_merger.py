"""
Attention Head Merger для GGUF моделей
Находит похожие головы внутри attn_q/k/v и мержит их.
Результат: модель с меньшим числом голов = меньше весов.
"""

import numpy as np
import re
import sys
from pathlib import Path

try:
    from gguf import GGUFReader, GGMLQuantizationType, GGUFWriter
    from gguf.quants import dequantize
except ImportError:
    print("pip install gguf")
    sys.exit(1)


# ─── Конфиг Qwen3-0.6B ───────────────────────────────────────────────
NUM_HEADS    = 16    # Q heads
NUM_KV_HEADS = 8     # K/V heads (GQA)
HEAD_DIM     = 64    # dim на голову
HIDDEN       = 1024
# attn_q: [num_heads * head_dim, hidden]  = [1024, 1024]  wait...
# проверим реально из модели


def load_tensors(path: str) -> tuple[dict, GGUFReader]:
    reader = GGUFReader(path)
    tensors = {}
    print("Декодируем тензоры...")
    for t in reader.tensors:
        try:
            f32 = dequantize(t.data, t.tensor_type).astype(np.float32).reshape(t.shape)
        except:
            try:
                f32 = t.data.astype(np.float32).reshape(t.shape)
            except:
                continue
        tensors[t.name] = f32
    print(f"Загружено: {len(tensors)} тензоров")
    return tensors, reader


def detect_config(tensors: dict) -> dict:
    """Определяем реальную конфигурацию голов из форм тензоров"""
    cfg = {}
    # Берём blk.0
    if "blk.0.attn_q.weight" in tensors:
        q = tensors["blk.0.attn_q.weight"]   # [q_total, hidden]
        k = tensors["blk.0.attn_k.weight"]   # [kv_total, hidden]
        v = tensors["blk.0.attn_v.weight"]
        o = tensors["blk.0.attn_output.weight"]  # [hidden, q_total]

        hidden    = q.shape[1]
        q_total   = q.shape[0]
        kv_total  = k.shape[0]

        # head_dim обычно 64 или 128
        # q_total = num_heads * head_dim
        # Пробуем угадать head_dim
        for hd in [128, 64, 32]:
            if q_total % hd == 0 and kv_total % hd == 0:
                head_dim  = hd
                num_q     = q_total  // hd
                num_kv    = kv_total // hd
                break

        cfg = {
            "hidden":    hidden,
            "head_dim":  head_dim,
            "num_q":     num_q,
            "num_kv":    num_kv,
            "q_shape":   q.shape,
            "k_shape":   k.shape,
        }
        print(f"\nКонфигурация модели:")
        print(f"  hidden={hidden}, head_dim={head_dim}")
        print(f"  Q heads={num_q}, KV heads={num_kv} (GQA ratio={num_q//num_kv})")
        print(f"  attn_q: {q.shape}, attn_k: {k.shape}, attn_output: {o.shape}")
    return cfg


def split_heads(W: np.ndarray, num_heads: int, head_dim: int) -> np.ndarray:
    """[total, hidden] -> [num_heads, head_dim, hidden]"""
    return W.reshape(num_heads, head_dim, W.shape[1])


def merge_two_heads(h1: np.ndarray, h2: np.ndarray, method="mean") -> np.ndarray:
    """Два веса головы [head_dim, hidden] -> один [head_dim, hidden]"""
    if method == "mean":
        return (h1 + h2) / 2.0
    elif method == "weighted":
        n1 = np.linalg.norm(h1)
        n2 = np.linalg.norm(h2)
        return (h1 * n1 + h2 * n2) / (n1 + n2 + 1e-8)
    elif method == "svd":
        # Берём первую SVD компоненту из стека двух голов
        stacked = np.vstack([h1, h2])  # [2*head_dim, hidden]
        U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
        # Берём top head_dim компонент
        return (U[:h1.shape[0], :] * S[None, :h1.shape[0]]) @ Vt[:h1.shape[0], :]
    return (h1 + h2) / 2.0


def cosine_matrix(heads: np.ndarray) -> np.ndarray:
    """heads: [N, head_dim, hidden] -> матрица сходства [N, N]"""
    N = heads.shape[0]
    flat = heads.reshape(N, -1).astype(np.float32)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    normed = flat / norms
    return normed @ normed.T


def find_best_pairs(sim_matrix: np.ndarray, n_pairs: int = 3) -> list:
    """Находим n_pairs самых похожих пар голов"""
    N = sim_matrix.shape[0]
    mat = sim_matrix.copy()
    np.fill_diagonal(mat, -1)
    pairs = []
    used = set()
    while len(pairs) < n_pairs:
        idx = np.unravel_index(np.argmax(mat), mat.shape)
        i, j = idx
        if i not in used and j not in used:
            pairs.append((int(i), int(j), float(mat[i, j])))
            used.add(i)
            used.add(j)
        mat[i, j] = mat[j, i] = -1
        if np.max(mat) < 0:
            break
    return pairs


def analyze_heads(tensors: dict, cfg: dict, n_layers: int = 5):
    """Анализируем похожесть голов по всем слоям"""
    print(f"\n{'='*65}")
    print("Анализ похожести attention голов:")
    print(f"{'='*65}")
    print(f"{'Слой':<8} {'Тип':<6} {'Пара':>10} {'Сходство':>10} {'Метод':>10}")
    print("-" * 50)

    all_results = []

    layer_indices = sorted(set(
        int(m.group(1))
        for name in tensors
        if (m := re.match(r"blk\.(\d+)\.", name))
    ))

    for layer in layer_indices[:n_layers]:
        for wtype, num_h in [("attn_q", cfg["num_q"]), ("attn_k", cfg["num_kv"]), ("attn_v", cfg["num_kv"])]:
            name = f"blk.{layer}.{wtype}.weight"
            if name not in tensors:
                continue

            W = tensors[name]
            heads = split_heads(W, num_h, cfg["head_dim"])
            sim   = cosine_matrix(heads)
            pairs = find_best_pairs(sim, n_pairs=1)

            for i, j, s in pairs:
                all_results.append({
                    "layer": layer, "type": wtype,
                    "i": i, "j": j, "sim": s,
                    "name": name, "W": W,
                    "num_heads": num_h,
                })
                print(f"  blk.{layer:<4} {wtype:<12} heads {i:>2}+{j:<2} sim={s:>8.4f}")

    all_results.sort(key=lambda x: -x["sim"])
    return all_results


def do_merge(tensors: dict, cfg: dict, results: list, n_merge: int = 2, method: str = "mean"):
    """
    Применяем слияние: убираем n_merge голов в самых похожих парах.
    Возвращает изменённые тензоры.
    """
    new_tensors = dict(tensors)  # копия
    merged_info = []

    for r in results[:n_merge]:
        name   = r["name"]
        W      = r["W"]
        num_h  = r["num_heads"]
        i, j   = r["i"], r["j"]

        heads = split_heads(W, num_h, cfg["head_dim"])

        # Мержим головы i и j -> одна голова
        merged_head = merge_two_heads(heads[i], heads[j], method)

        # Убираем голову j, голову i заменяем merged
        new_heads = []
        for h_idx in range(num_h):
            if h_idx == j:
                continue  # удаляем
            elif h_idx == i:
                new_heads.append(merged_head)
            else:
                new_heads.append(heads[h_idx])

        new_W = np.stack(new_heads, axis=0).reshape(-1, W.shape[1])
        new_tensors[name] = new_W

        save_mb = (W.nbytes - new_W.nbytes) / 1024**2
        merged_info.append({
            "name": name,
            "old_shape": W.shape, "new_shape": new_W.shape,
            "save_mb": save_mb,
        })
        print(f"  Merged: {name}")
        print(f"    {W.shape} -> {new_W.shape}  (-{save_mb:.2f} MB)")

    return new_tensors, merged_info


def test_merge_quality(tensors: dict, new_tensors: dict, cfg: dict, layer: int = 0):
    """
    Быстрая проверка: прогоняем случайный вектор через оригинальный
    и через merged attn_q, смотрим насколько отличается.
    """
    name = f"blk.{layer}.attn_q.weight"
    if name not in tensors or name not in new_tensors:
        return

    W_orig = tensors[name]
    W_new  = new_tensors[name]

    x = np.random.randn(50, cfg["hidden"]).astype(np.float32)

    out_orig = x @ W_orig.T   # [50, q_total]
    out_new  = x @ W_new.T    # [50, q_total - head_dim]

    # Сравниваем только общие размерности
    min_dim = min(out_orig.shape[1], out_new.shape[1])
    mse = np.mean((out_orig[:, :min_dim] - out_new[:, :min_dim]) ** 2)
    rel = mse / (np.mean(out_orig[:, :min_dim] ** 2) + 1e-8)

    print(f"\nКачество слияния (blk.{layer}.attn_q):")
    print(f"  Оригинал выход: {out_orig.shape}")
    print(f"  После мержа:    {out_new.shape}")
    print(f"  MSE:  {mse:.6f}")
    print(f"  Отн. ошибка: {rel*100:.2f}%")


def main():
    path = r"F:\12\Sovenok-Hacker\Qwen3-0.6B-Q4_0-GGUF\qwen3-0.6b-q4_0.gguf"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    if not Path(path).exists():
        print(f"Файл не найден: {path}")
        sys.exit(1)

    tensors, reader = load_tensors(path)
    cfg = detect_config(tensors)

    if not cfg:
        print("Не удалось определить конфигурацию модели")
        sys.exit(1)

    # Анализируем первые 10 слоёв
    results = analyze_heads(tensors, cfg, n_layers=10)

    print(f"\n{'='*65}")
    print(f"ТОП-5 самых похожих пар голов по всей модели:")
    print(f"{'='*65}")
    for r in results[:5]:
        print(f"  blk.{r['layer']}.{r['type']} heads {r['i']}+{r['j']}  sim={r['sim']:.4f}")

    # Мержим топ-2 пары
    print(f"\nМержим топ-2 пары голов (метод: mean)...")
    new_tensors, info = do_merge(tensors, cfg, results, n_merge=2, method="mean")

    # Проверяем качество
    test_merge_quality(tensors, new_tensors, cfg, layer=results[0]["layer"])

    total_save = sum(i["save_mb"] for i in info)
    print(f"\nИтого сэкономлено: {total_save:.2f} MB")
    print(f"\nЕсли мержить по 2 головы на всех 28 слоях:")
    print(f"  ~{total_save * 28 / 2:.0f} MB экономии")
    print(f"\nДля сохранения в новый GGUF нужен отдельный шаг (скажи если нужен)")


if __name__ == "__main__":
    main()
