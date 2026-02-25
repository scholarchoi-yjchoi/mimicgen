#!/usr/bin/env python3
"""
HDF5 데이터 구조 심층 분석 스크립트
- annotated_dataset.hdf5 (입력)
- generated_dataset.hdf5 (출력 - 성공)
- generated_dataset_failed.hdf5 (출력 - 실패)
"""

import h5py
import numpy as np
import os
import sys

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "analysis_results", "hdf5_analysis.txt")

FILES = {
    "annotated": os.path.join(DATASETS_DIR, "annotated_dataset.hdf5"),
    "generated": os.path.join(DATASETS_DIR, "generated_dataset.hdf5"),
    "failed": os.path.join(DATASETS_DIR, "generated_dataset_failed.hdf5"),
}


class Logger:
    """stdout와 파일에 동시 출력"""
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, "w", encoding="utf-8")

    def write(self, msg=""):
        print(msg)
        self.file.write(msg + "\n")

    def close(self):
        self.file.close()


def print_separator(log, char="=", length=80):
    log.write(char * length)


def analyze_structure(log, f, label):
    """HDF5 파일의 전체 구조를 재귀적으로 출력"""
    print_separator(log)
    log.write(f"[구조 분석] {label}")
    log.write(f"파일 크기: {os.path.getsize(f.filename) / 1024:.1f} KB")
    print_separator(log)

    # Root attributes
    log.write("\n--- Root Attributes ---")
    if len(f.attrs) == 0:
        log.write("  (없음)")
    for attr_name in f.attrs:
        val = f.attrs[attr_name]
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        log.write(f"  @{attr_name} = {val}")

    # Recursive structure
    log.write("\n--- 전체 구조 ---")

    def visit_item(name, obj):
        indent = "  " * (name.count("/") + 1)
        if isinstance(obj, h5py.Dataset):
            dtype_str = str(obj.dtype)
            shape_str = str(obj.shape)
            # 값 미리보기
            preview = ""
            if obj.size > 0:
                if obj.dtype.kind in ("f", "i", "u"):
                    data = obj[()]
                    if data.ndim == 0:
                        preview = f" = {data}"
                    elif data.size <= 10:
                        preview = f" = {data}"
                    else:
                        preview = f" min={np.min(data):.6g}, max={np.max(data):.6g}, mean={np.mean(data):.6g}"
                elif obj.dtype.kind == "S" or obj.dtype.kind == "O":
                    try:
                        val = obj[()]
                        if isinstance(val, bytes):
                            val = val.decode("utf-8")
                        preview = f' = "{val}"' if len(str(val)) < 200 else f' = "{str(val)[:200]}..."'
                    except Exception:
                        preview = " (읽기 실패)"
            log.write(f"{indent}{name}  [{dtype_str}] {shape_str}{preview}")
            # Dataset attributes
            for attr_name in obj.attrs:
                attr_val = obj.attrs[attr_name]
                if isinstance(attr_val, bytes):
                    attr_val = attr_val.decode("utf-8")
                log.write(f"{indent}  @{attr_name} = {attr_val}")
        elif isinstance(obj, h5py.Group):
            n_children = len(obj)
            log.write(f"{indent}{name}/  (Group, {n_children} children)")
            for attr_name in obj.attrs:
                attr_val = obj.attrs[attr_name]
                if isinstance(attr_val, bytes):
                    attr_val = attr_val.decode("utf-8")
                log.write(f"{indent}  @{attr_name} = {attr_val}")

    f.visititems(visit_item)
    log.write("")


def analyze_demos(log, f, label):
    """각 데모의 상세 통계"""
    print_separator(log, "-")
    log.write(f"[데모별 통계] {label}")
    print_separator(log, "-")

    if "data" not in f:
        log.write("  'data' 그룹 없음")
        return

    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
    log.write(f"총 데모 수: {len(demos)}")

    for demo_name in demos:
        demo = f["data"][demo_name]
        log.write(f"\n  === {demo_name} ===")

        # Demo-level attributes
        for attr_name in demo.attrs:
            attr_val = demo.attrs[attr_name]
            if isinstance(attr_val, bytes):
                attr_val = attr_val.decode("utf-8")
            log.write(f"    @{attr_name} = {attr_val}")

        # All keys in this demo
        all_keys = []

        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                all_keys.append(name)

        demo.visititems(collect)

        for key in sorted(all_keys):
            ds = demo[key]
            shape = ds.shape
            dtype = ds.dtype
            line = f"    {key}: shape={shape}, dtype={dtype}"

            if dtype.kind in ("f", "i", "u") and ds.size > 0:
                data = ds[()]
                if data.ndim >= 1:
                    line += f", min={np.min(data):.6g}, max={np.max(data):.6g}, mean={np.mean(data):.6g}"
                    if data.ndim == 2:
                        line += f"\n      first_row={data[0]}"
                        line += f"\n      last_row={data[-1]}"
                else:
                    line += f", value={data}"
            elif dtype.kind in ("S", "O"):
                try:
                    val = ds[()]
                    if isinstance(val, bytes):
                        val = val.decode("utf-8")
                    line += f', value="{val}"' if len(str(val)) < 100 else f', value="{str(val)[:100]}..."'
                except Exception:
                    pass

            log.write(line)

    log.write("")


def analyze_action_semantics(log, f, label):
    """Action 차원별 의미 분석 (IK-Rel 확인)"""
    print_separator(log, "-")
    log.write(f"[Action 의미 분석] {label}")
    print_separator(log, "-")

    if "data" not in f:
        return

    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))

    # 모든 데모의 actions를 수집
    all_actions = []
    for demo_name in demos:
        if "actions" in f["data"][demo_name]:
            all_actions.append(f["data"][demo_name]["actions"][:])

    if not all_actions:
        log.write("  actions 데이터 없음")
        return

    all_actions = np.concatenate(all_actions, axis=0)
    log.write(f"총 action 프레임: {all_actions.shape[0]}")
    log.write(f"Action 차원: {all_actions.shape[1]}")

    # IK-Rel 가능성: [dx, dy, dz, dqx, dqy, dqz, gripper]
    dim_names_guess = ["dx (or joint_0)", "dy (or joint_1)", "dz (or joint_2)",
                       "dqx (or joint_3)", "dqy (or joint_4)", "dqz (or joint_5)",
                       "gripper (or joint_6)"]

    log.write("\n차원별 통계:")
    for dim in range(all_actions.shape[1]):
        a = all_actions[:, dim]
        unique_count = len(np.unique(np.round(a, 4)))
        name_guess = dim_names_guess[dim] if dim < len(dim_names_guess) else f"dim_{dim}"
        log.write(
            f"  dim {dim} ({name_guess}):"
            f" min={a.min():.6f}, max={a.max():.6f},"
            f" mean={a.mean():.6f}, std={a.std():.6f},"
            f" unique_values(4dp)={unique_count}"
        )

    # 마지막 차원이 그리퍼인지 확인 (이산적 값이면 그리퍼일 가능성)
    last_dim = all_actions[:, -1]
    unique_last = np.unique(np.round(last_dim, 2))
    if len(unique_last) <= 10:
        log.write(f"\n  마지막 차원 고유 값 (rounded 2dp): {unique_last}")
        log.write("  → 이산적 값 → 그리퍼 명령일 가능성 높음")
    else:
        log.write(f"\n  마지막 차원 고유 값 수: {len(unique_last)} → 연속 값")

    log.write("")


def analyze_env_args(log, f, label):
    """env_args 메타데이터 분석"""
    print_separator(log, "-")
    log.write(f"[env_args 분석] {label}")
    print_separator(log, "-")

    if "env_args" not in f:
        # Top-level keys 중 env_args가 아닌 것도 확인
        log.write("  'env_args' 그룹 없음")
        log.write(f"  Top-level keys: {list(f.keys())}")
        return

    env_args = f["env_args"]

    def print_group(group, prefix=""):
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Group):
                log.write(f"  {prefix}{key}/ (Group)")
                for attr_name in item.attrs:
                    attr_val = item.attrs[attr_name]
                    if isinstance(attr_val, bytes):
                        attr_val = attr_val.decode("utf-8")
                    log.write(f"  {prefix}  @{attr_name} = {attr_val}")
                print_group(item, prefix + "  ")
            elif isinstance(item, h5py.Dataset):
                val = item[()]
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                log.write(f"  {prefix}{key}: {val}")

    print_group(env_args)

    for attr_name in env_args.attrs:
        attr_val = env_args.attrs[attr_name]
        if isinstance(attr_val, bytes):
            attr_val = attr_val.decode("utf-8")
        log.write(f"  @{attr_name} = {attr_val}")

    log.write("")


def analyze_annotations(log, f, label):
    """MimicGen subtask annotation 분석"""
    print_separator(log, "-")
    log.write(f"[Annotation/Subtask 분석] {label}")
    print_separator(log, "-")

    if "data" not in f:
        return

    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))[:3]  # 처음 3개만

    annotation_keys = [
        "datagen_info", "subtask_labels", "subtask_indices",
        "subtask_boundaries", "subtask_signals", "annotation",
        "subtask_term_signals", "intv_labels",
    ]

    found_any = False
    for demo_name in demos:
        demo = f["data"][demo_name]
        log.write(f"\n  === {demo_name} ===")

        # 모든 키를 재귀적으로 수집
        all_keys = []

        def collect(name, obj):
            all_keys.append((name, type(obj).__name__))

        demo.visititems(collect)

        for key in annotation_keys:
            if key in demo:
                found_any = True
                item = demo[key]
                if isinstance(item, h5py.Dataset):
                    val = item[()]
                    if isinstance(val, bytes):
                        val = val.decode("utf-8")
                    log.write(f"    {key}: shape={item.shape}, dtype={item.dtype}")
                    if item.size < 50:
                        log.write(f"      value: {val}")
                elif isinstance(item, h5py.Group):
                    log.write(f"    {key}/ (Group, keys={list(item.keys())})")
                    for sub_k in item:
                        sub_v = item[sub_k]
                        if isinstance(sub_v, h5py.Dataset):
                            sub_val = sub_v[()]
                            if isinstance(sub_val, bytes):
                                sub_val = sub_val.decode("utf-8")
                            log.write(f"      {sub_k}: shape={sub_v.shape}, dtype={sub_v.dtype}")
                            if sub_v.size < 50:
                                log.write(f"        value: {sub_val}")

        # 알려진 키 외의 특이한 키도 출력
        known_keys = {"actions", "states", "initial_state", "obs"}
        unknown_top = [name for name, typ in all_keys if "/" not in name and name not in known_keys]
        if unknown_top:
            log.write(f"    기타 top-level 키: {unknown_top}")

    if not found_any:
        log.write("  알려진 annotation 키를 찾지 못함")
        log.write("  (데이터에 직접적인 subtask label이 없을 수 있음)")

    log.write("")


def compare_input_output(log, f_in, f_out, f_fail):
    """입력/출력/실패 파일 비교"""
    print_separator(log)
    log.write("[입출력 비교 분석]")
    print_separator(log)

    # 데모 수
    n_input = len(f_in["data"].keys()) if "data" in f_in else 0
    n_success = len(f_out["data"].keys()) if "data" in f_out else 0
    n_failed = len(f_fail["data"].keys()) if "data" in f_fail else 0
    n_total = n_success + n_failed

    log.write(f"입력 데모 수: {n_input}")
    log.write(f"출력 성공 데모 수: {n_success}")
    log.write(f"출력 실패 데모 수: {n_failed}")
    log.write(f"총 시도: {n_total}")
    if n_total > 0:
        log.write(f"성공률: {n_success / n_total * 100:.1f}%")
    if n_input > 0:
        log.write(f"증폭 비율: {n_success}x (성공) / {n_total}x (총시도) from {n_input} input demos")

    # 구조 비교
    log.write("\n--- 필드 비교 ---")

    def get_all_dataset_keys(demo_group):
        keys = set()

        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                keys.add(name)

        demo_group.visititems(collect)
        return keys

    if "data" in f_in and len(f_in["data"]) > 0:
        input_demo = f_in["data"][list(f_in["data"].keys())[0]]
        input_keys = get_all_dataset_keys(input_demo)
    else:
        input_keys = set()

    if "data" in f_out and len(f_out["data"]) > 0:
        output_demo = f_out["data"][list(f_out["data"].keys())[0]]
        output_keys = get_all_dataset_keys(output_demo)
    else:
        output_keys = set()

    if "data" in f_fail and len(f_fail["data"]) > 0:
        failed_demo = f_fail["data"][list(f_fail["data"].keys())[0]]
        failed_keys = get_all_dataset_keys(failed_demo)
    else:
        failed_keys = set()

    log.write(f"입력 데모 필드: {sorted(input_keys)}")
    log.write(f"출력(성공) 데모 필드: {sorted(output_keys)}")
    log.write(f"출력(실패) 데모 필드: {sorted(failed_keys)}")

    added = output_keys - input_keys
    removed = input_keys - output_keys
    if added:
        log.write(f"\n출력에 추가된 필드: {sorted(added)}")
    if removed:
        log.write(f"출력에서 제거된 필드: {sorted(removed)}")
    if not added and not removed:
        log.write("\n입출력 필드 동일")

    # Trajectory 길이 비교
    log.write("\n--- Trajectory 길이 비교 ---")

    def get_trajectory_lengths(f_handle):
        lengths = []
        if "data" in f_handle:
            for demo_name in f_handle["data"]:
                demo = f_handle["data"][demo_name]
                if "actions" in demo:
                    lengths.append(demo["actions"].shape[0])
        return lengths

    input_lengths = get_trajectory_lengths(f_in)
    output_lengths = get_trajectory_lengths(f_out)
    failed_lengths = get_trajectory_lengths(f_fail)

    for name, lengths in [("입력", input_lengths), ("출력(성공)", output_lengths), ("출력(실패)", failed_lengths)]:
        if lengths:
            log.write(
                f"  {name}: count={len(lengths)}, "
                f"min={min(lengths)}, max={max(lengths)}, "
                f"mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}"
            )
        else:
            log.write(f"  {name}: 데이터 없음")

    # Action 범위 비교
    log.write("\n--- Action 범위 비교 ---")

    def get_all_actions(f_handle):
        actions_list = []
        if "data" in f_handle:
            for demo_name in f_handle["data"]:
                if "actions" in f_handle["data"][demo_name]:
                    actions_list.append(f_handle["data"][demo_name]["actions"][:])
        if actions_list:
            return np.concatenate(actions_list, axis=0)
        return None

    for name, f_handle in [("입력", f_in), ("출력(성공)", f_out), ("출력(실패)", f_fail)]:
        actions = get_all_actions(f_handle)
        if actions is not None:
            log.write(f"  {name}: shape={actions.shape}")
            for dim in range(actions.shape[1]):
                log.write(
                    f"    dim {dim}: [{actions[:, dim].min():.6f}, {actions[:, dim].max():.6f}]"
                    f" mean={actions[:, dim].mean():.6f}"
                )

    # Root attributes 비교
    log.write("\n--- Root Attributes 비교 ---")
    for name, fh in [("입력", f_in), ("출력(성공)", f_out), ("출력(실패)", f_fail)]:
        log.write(f"  {name}:")
        for attr in fh.attrs:
            val = fh.attrs[attr]
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            log.write(f"    @{attr} = {val}")

    log.write("")


def analyze_franka_dependency(log, f, label):
    """Franka 로봇 종속성 식별"""
    print_separator(log, "-")
    log.write(f"[Franka 종속성 식별] {label}")
    print_separator(log, "-")

    # Action 차원 = 로봇 DoF
    if "data" in f and len(f["data"]) > 0:
        demo = f["data"][list(f["data"].keys())[0]]
        if "actions" in demo:
            action_dim = demo["actions"].shape[1]
            log.write(f"  Action 차원: {action_dim} (Franka 7-DoF → G1은 다른 DoF)")

        if "obs" in demo:
            obs = demo["obs"]
            for key in obs:
                if isinstance(obs[key], h5py.Dataset):
                    log.write(f"  obs/{key}: shape={obs[key].shape} (dim={obs[key].shape[-1] if obs[key].ndim > 1 else obs[key].shape[0]})")

    # env_name 확인
    for attr in f.attrs:
        val = f.attrs[attr]
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        if "franka" in str(val).lower() or "panda" in str(val).lower():
            log.write(f"  Root @{attr} = {val}  ← FRANKA 참조!")

    if "env_args" in f:
        def check_franka(group, prefix=""):
            for key in group:
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    val = item[()]
                    if isinstance(val, bytes):
                        val = val.decode("utf-8")
                    val_str = str(val)
                    if "franka" in val_str.lower() or "panda" in val_str.lower():
                        log.write(f"  env_args/{prefix}{key} = {val}  ← FRANKA 참조!")
                elif isinstance(item, h5py.Group):
                    check_franka(item, prefix + key + "/")

        check_franka(f["env_args"])

    log.write("")


def main():
    log = Logger(OUTPUT_FILE)
    log.write("=" * 80)
    log.write("HDF5 데이터 구조 심층 분석 결과")
    log.write(f"분석 일시: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.write("=" * 80)

    handles = {}
    for label, path in FILES.items():
        if os.path.exists(path):
            handles[label] = h5py.File(path, "r")
            log.write(f"\n  {label}: {path} ({os.path.getsize(path) / 1024:.1f} KB)")
        else:
            log.write(f"\n  {label}: {path} (파일 없음!)")

    log.write("")

    # 각 파일별 분석
    for label, f in handles.items():
        analyze_structure(log, f, label)
        analyze_demos(log, f, label)
        analyze_action_semantics(log, f, label)
        analyze_env_args(log, f, label)
        analyze_annotations(log, f, label)
        analyze_franka_dependency(log, f, label)

    # 입출력 비교
    if "annotated" in handles and "generated" in handles and "failed" in handles:
        compare_input_output(log, handles["annotated"], handles["generated"], handles["failed"])

    # 정리
    for f in handles.values():
        f.close()

    log.write("\n" + "=" * 80)
    log.write("분석 완료")
    log.write("=" * 80)
    log.close()
    print(f"\n결과 저장: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
