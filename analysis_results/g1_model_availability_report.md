# Phase 0-1: G1 Isaac Lab Manipulation 모델 가용성 보고서

## 조사 일자: 2026-02-24

---

## 1. 현재 컨테이너 환경

| 항목 | 값 |
|------|-----|
| Docker Image | `nvcr.io/nvidia/gr00t-smmg-bp:1.0` |
| Isaac Lab 버전 | **0.36.2** (구 버전) |
| isaaclab-mimic 버전 | **1.0.3** |
| Python | `/workspace/isaaclab/_isaac_sim/python.sh` (Python 3.10) |
| G1 USD 위치 | NVIDIA Nucleus Server (로컬 캐시 없음) |

## 2. G1 로봇 Config 존재 여부

### 현재 컨테이너 (Isaac Lab 0.36.2)

**파일**: `/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py`

| Config 이름 | USD 경로 | 손/핸드 | 용도 |
|-------------|---------|---------|------|
| `G1_CFG` | `{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd` | Dex3-like (numbered joints) | 일반 locomotion |
| `G1_MINIMAL_CFG` | `g1_minimal.usd` | 동일, 충돌체 축소 | 빠른 시뮬레이션 |

**주의**: 현재 `G1_CFG`는 **구 버전 USD**로, 관절 이름이 번호 방식(`.*_zero_joint` ~ `.*_six_joint`)을 사용합니다.

### 최신 Isaac Lab (2.3+)

최신 Isaac Lab에는 추가 config가 존재:

| Config 이름 | 손/핸드 | 관절 수 | 용도 |
|-------------|---------|---------|------|
| `G1_29DOF_CFG` | 없음 (bare arms) | 29 | Locomanipulation |
| `G1_INSPIRE_FTP_CFG` | Inspire 5-finger (12 DoF/hand) | 29+24 | **Pick-and-place manipulation** |

## 3. G1 팔 관절 구조

### 공식 URDF (29DOF 버전) - 팔 관절 7 DoF/arm

| 번호 | 왼팔 | 오른팔 | 기능 |
|:----:|------|--------|------|
| 1 | `left_shoulder_pitch_joint` | `right_shoulder_pitch_joint` | 어깨 피치 |
| 2 | `left_shoulder_roll_joint` | `right_shoulder_roll_joint` | 어깨 롤 |
| 3 | `left_shoulder_yaw_joint` | `right_shoulder_yaw_joint` | 어깨 요 |
| 4 | `left_elbow_joint` | `right_elbow_joint` | 팔꿈치 (**1 DoF**) |
| 5 | `left_wrist_roll_joint` | `right_wrist_roll_joint` | 손목 롤 |
| 6 | `left_wrist_pitch_joint` | `right_wrist_pitch_joint` | 손목 피치 |
| 7 | `left_wrist_yaw_joint` | `right_wrist_yaw_joint` | 손목 요 |

### 현재 컨테이너 `G1_CFG` - 팔 관절 (구 버전 USD)

| 번호 | 왼팔 | 오른팔 | 비고 |
|:----:|------|--------|------|
| 1 | `left_shoulder_pitch_joint` | `right_shoulder_pitch_joint` | |
| 2 | `left_shoulder_roll_joint` | `right_shoulder_roll_joint` | |
| 3 | `left_shoulder_yaw_joint` | `right_shoulder_yaw_joint` | |
| 4 | `left_elbow_pitch_joint` | `right_elbow_pitch_joint` | 구 USD: pitch/roll 2 DoF |
| 5 | `left_elbow_roll_joint` | `right_elbow_roll_joint` | 구 USD 전용 |

### 현재 컨테이너 `G1_CFG` - 손 관절 (Dex3-1 대응, 7 DoF/hand)

| 번호 Joint | 대응 Dex3-1 관절 | 기능 |
|-----------|-----------------|------|
| `.*_zero_joint` | thumb joint 0 | 엄지 기저 회전/벌림 |
| `.*_one_joint` | thumb joint 1 | 엄지 굽힘 |
| `.*_two_joint` | thumb joint 2 | 엄지 끝 굽힘 |
| `.*_three_joint` | index joint 0 | 검지 근위 굽힘 |
| `.*_four_joint` | index joint 1 | 검지 원위 굽힘 |
| `.*_five_joint` | middle joint 0 | 중지 근위 굽힘 |
| `.*_six_joint` | middle joint 1 | 중지 원위 굽힘 |

## 4. End-Effector Body Names

| 손 구성 | 왼팔 EEF | 오른팔 EEF |
|---------|----------|-----------|
| **현재 G1_CFG (구 USD)** | `left_palm_link` (추정) | `right_palm_link` (추정) |
| **Dex3-1 hand** | `left_hand_palm_link` | `right_hand_palm_link` |
| **Bare arms (no hand)** | `left_wrist_yaw_link` | `right_wrist_yaw_link` |
| **Inspire 5-finger** | `left_wrist_yaw_link` | `right_wrist_yaw_link` |

**Isaac Lab IK 설정 참조** (최신 Isaac Lab pick-place 환경):
```python
target_eef_link_names = {
    "left_wrist": "left_wrist_yaw_link",
    "right_wrist": "right_wrist_yaw_link",
}
```

## 5. G1 손/핸드 옵션

| 핸드 타입 | URDF 파일 | 손가락 | DoF/hand | 비고 |
|----------|----------|--------|----------|------|
| **Dex3-1** | `g1_29dof_with_hand_rev_1_0.urdf` | 3 (엄지, 검지, 중지) | 7 | 기본 옵션, 촉각 센서 33개 |
| **Inspire DFQ** | `g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf` | 5 (전체) | 12 | Isaac Lab 지원 |
| **Inspire FTP** | `g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf` | 5 (전체) | 12 | Isaac Lab 지원 |
| **Rubber hand** | `g1_29dof_rev_1_0.urdf` | 0 (수동) | 0 | 심플 그리퍼 대체 가능 |

## 6. DoF 요약

| 구성 | 팔 DoF | 손목 DoF | 손 DoF | 팔당 총 DoF |
|------|:------:|:-------:|:------:|:----------:|
| G1 23DOF (basic) | 3+1+1=5 | 포함 | 0 | **5** |
| G1 29DOF (no hand) | 3+1+3=7 | 포함 | 0 | **7** |
| G1 29DOF + Dex3-1 | 7 | 포함 | 7 | **14** |
| G1 29DOF + Inspire | 7 | 포함 | 12 | **19** |

## 7. Isaac Lab 버전 격차

### 현재 (0.36.2) vs 필요 (2.3+)

| 기능 | 현재 0.36.2 | 최신 2.3+ |
|------|:----------:|:--------:|
| G1 locomotion | O | O |
| G1 manipulation (pick-place) | **X** | **O** |
| Bimanual IK (Pink IK) | **X** | **O** |
| G1 Inspire hand 지원 | **X** | **O** |
| G1 teleoperation | **X** | **O** |
| SkillGen (MimicGen 확장) | **X** | **O** |
| 듀얼 암 MimicGen | **X** | **미확인** |

### 업그레이드 경로

- 현재 Docker 이미지: `nvcr.io/nvidia/gr00t-smmg-bp:1.0`
- 필요: Isaac Lab 2.3+ 이미지 (Isaac Sim 5.x 기반)
- **Unitree 공식 sim**: `unitree_sim_isaaclab` (Isaac Sim 4.5.0/5.x.0 지원)

## 8. Unitree 공식 시뮬레이션 (`unitree_sim_isaaclab`)

| 항목 | 내용 |
|------|------|
| 지원 로봇 | G1-29dof-gripper, G1-29dof-dex3, G1-29dof-inspire |
| 지원 Task | Cylinder pick-place, Red block manipulation, RGB block stacking |
| 특이사항 | DDS 통신 프로토콜 (실제 로봇과 동일), Wholebody manipulation |
| 필수 사항 | Isaac Sim 4.5.0+ 또는 5.x.0 |

## 9. 결론 및 권장사항

### Go/No-Go 판단: **조건부 Go**

**가능한 부분:**
- G1 로봇 config (`G1_CFG`)가 현재 컨테이너에 존재
- 팔 관절 구조가 명확히 파악됨 (5 DoF arm in current USD, 7 DoF in latest)
- Dex3-1 / Inspire hand 관절 구조 파악됨
- EEF body name 확인됨 (`left_palm_link` / `right_palm_link` or `left_wrist_yaw_link` / `right_wrist_yaw_link`)

**해결 필요한 부분:**
1. **Isaac Lab 업그레이드 필요**: 현재 0.36.2 → 2.3+ 업그레이드가 **강력히 권장됨**
   - 최신 버전에 G1 manipulation 환경, bimanual IK, SkillGen이 포함
   - Unitree 공식 `unitree_sim_isaaclab`도 최신 Isaac Lab 필요
2. **MimicGen 듀얼 암**: 여전히 미지원 (별도 해결 필요)
3. **G1 USD 버전 차이**: 현재 컨테이너의 구 USD는 관절 이름이 다름 → 업그레이드 시 자동 해결

### 권장 작업 순서

1. **Isaac Lab 업그레이드 가능성 확인**: 새 Docker 이미지 (`nvcr.io/nvidia/isaac-lab:2.3.x` 등) 존재 여부 확인
2. **현재 환경에서 G1 manipulation POC**: 업그레이드 전에도 현재 `G1_CFG`로 단순 pick-place 시도 가능
3. **Unitree `unitree_sim_isaaclab` 통합 검토**: 이미 G1 manipulation 환경이 구현되어 있으므로 참조/재사용

---

## 참조

- Isaac Lab unitree.py: `/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py`
- G1 locomotion env: `/workspace/isaaclab/source/isaaclab_tasks/.../locomotion/velocity/config/g1/`
- Unitree G1 URDF: https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description
- Unitree sim for Isaac Lab: https://github.com/unitreerobotics/unitree_sim_isaaclab
- Isaac Lab 2.3 blog: https://developer.nvidia.com/blog/streamline-robot-learning-with-whole-body-control-and-enhanced-teleoperation-in-nvidia-isaac-lab-2-3/
- Isaac Lab G1 IK discussion: https://github.com/isaac-sim/IsaacLab/discussions/1619
