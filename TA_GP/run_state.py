"""
Chống mất data âm thầm + resume cho vòng lặp defense.

VẤN ĐỀ ĐANG SỬA
---------------
Bản gốc của main_defense làm thế này:

    except Exception as e:
        print(e)
        continue

Một lỗi 429 (rate limit) ở bất kỳ agent nào -> BỎ NGUYÊN sample đó, vòng lặp
chạy tiếp, file kết quả vẫn ghi, exit code vẫn 0. Hậu quả: CSV có số liệu tính
trên n < samples mà không ai biết, và n khác nhau giữa các backbone -> so sánh
lệch mà không phát hiện được. Nguy hiểm hơn crash, vì crash thì còn biết mà sửa.

Bằng chứng đã xảy ra thật: trong evaluation_results (2).csv, num_samples của
topology `tree` dao động 55–60 trong khi các topology khác là 58–60.

CÁCH XỬ LÝ
----------
1. Đếm số sample bị bỏ, in ra SAMPLES_OK / SAMPLES_SKIPPED.
2. Vượt ngưỡng MAX_SKIP_RATIO (default 10%) -> exit code 2, để run_all.sh
   coi run đó là FAILED thay vì ghi nhận kết quả rác.
3. Ghi từng sample vào .partial.jsonl ngay khi xong -> chạy lại thì bỏ qua
   phần đã có (mất điện / OOM / 429 kéo dài không mất tiến độ).
"""

import json
import os
import sys


class RunState:
    def __init__(self, path_nd, path_wd, max_skip_ratio=None):
        self.path_nd = path_nd
        self.path_wd = path_wd
        self.partial = path_nd + ".partial.jsonl"
        self.max_skip_ratio = (
            float(os.getenv("MAX_SKIP_RATIO", "0.1"))
            if max_skip_ratio is None else max_skip_ratio
        )
        self.nd, self.wd = [], []
        self.skipped = []
        self._done = set()
        self._load_partial()

    def _load_partial(self):
        if not os.path.exists(self.partial):
            return
        bad = 0
        with open(self.partial) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1          # dòng cuối ghi dở do bị kill -> bỏ
                    continue
                self._done.add(rec["key"])
                self.nd.append(rec["nd"])
                self.wd.append(rec["wd"])
        print(f"RESUMED: đã có {len(self._done)} sample trong {self.partial}"
              + (f" ({bad} dòng hỏng bị bỏ)" if bad else ""))

    def is_done(self, key) -> bool:
        return key in self._done

    def record(self, key, d_nd, d_wd):
        self.nd.append(d_nd)
        self.wd.append(d_wd)
        self._done.add(key)
        with open(self.partial, "a") as f:      # append ngay -> resume được
            f.write(json.dumps({"key": key, "nd": d_nd, "wd": d_wd}) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def record_skip(self, key, err):
        self.skipped.append({"key": key, "err": repr(err)})
        print(f"⚠️  SKIP sample {key}: {err!r}", flush=True)

    def finalize(self):
        """Ghi file cuối. Exit 2 nếu tỉ lệ mất sample vượt ngưỡng."""
        n_ok, n_skip = len(self.nd), len(self.skipped)
        total = n_ok + n_skip

        with open(self.path_nd, "w") as f:
            json.dump(self.nd, f, indent=None)
        with open(self.path_wd, "w") as f:
            json.dump(self.wd, f, indent=None)

        print(f"SAMPLES_OK: {n_ok}")
        print(f"SAMPLES_SKIPPED: {n_skip}")

        if total and n_skip / total > self.max_skip_ratio:
            print(f"❌ Bỏ {n_skip}/{total} sample "
                  f"({n_skip/total:.1%}) > ngưỡng {self.max_skip_ratio:.0%} "
                  f"— KẾT QUẢ KHÔNG DÙNG ĐƯỢC.")
            for s in self.skipped[:5]:
                print(f"     - sample {s['key']}: {s['err'][:160]}")
            sys.exit(2)

        if os.path.exists(self.partial):
            os.remove(self.partial)     # xong sạch thì bỏ file tạm

        print(f"NO_DEFENSE_FILE: {self.path_nd}")
        print(f"DEFENSE_FILE: {self.path_wd}")
