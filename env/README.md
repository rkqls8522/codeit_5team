# 개발 환경 구축 가이드 (HealthIt 5팀)

## 0. [Windows사용자 필수] 사전 준비

**윈도우 사용자**는 오류 방지를 위해 가장 먼저 수행해야 합니다. (맥 사용자는 건너뛰세요)

- **Visual C++ 재배포 가능 패키지 설치**: [다운로드 링크](https://aka.ms/vs/17/release/vc_redist.x64.exe)를 눌러 설치 후 반드시 **재부팅** 해주세요.
- _이 과정을 생략하면 `ImportError`나 `WinError 127`이 발생할 수 있습니다._

---

## 1. 가상환경 생성

프로젝트 폴더에서 터미널을 열고 `environment.yml` 파일을 기반으로 환경을 생성합니다.

```bash
conda env create -f env/environment.yml
```

---

## 2. 가상환경 실행

생성된 가상환경을 활성화합니다.

```bash
conda activate healthit_5team
```

---

## 3. PyTorch 설치 (PC 사양에 맞춰 선택)

GPU 사용을 위해 PyTorch는 별도로 설치합니다. 본인 사양에 맞는 명령어 **하나만** 실행하세요.

### 3-1. Windows 사용자 + NVIDIA GPU 보유자

**만약 GPU가 있지만 CPU를 사용하실 예정이시거나, 간단하게 진행하고 싶으시면 3-2로 이동하세요.**

최신 호환성을 위해 **CUDA 12.1** 버전을 설치합니다.

(기존 캐시 충돌 방지를 위해 강제 재설치 옵션을 포함했습니다.)

```bash
pip install --force-reinstall --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

참고: 만약 구형 GPU(GTX 10/RTX 20 시리즈 등)를 사용한다면 CUDA 11.8 버전을 시도하세요.

```bash
pip install --force-reinstall --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

GPU사용 준비가 완료되었는지 확인하기 위해 아래 명령어를 복사해 터미널에 입력하세요.

```bash
python -c "import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'; import torch; print(f'GPU 가능 여부: {torch.cuda.is_available()}'); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None');"
```

- **결과가 `True`가 나오면 모델 학습 및 추론 시 GPU가 사용될 것이고, 그렇지 않다면 CPU가 사용될 것입니다!**

### 3-2. Mac(M1/M2) 사용자 또는 GPU가 없는 경우

기본 버전을 설치합니다.

```bash
pip install torch torchvision
```

---

## 4. 트러블슈팅

### 🚨 발생할 수 있는 에러와 해결법

**Q1. `OMP: Error #15` (라이브러리 중복 에러)가 뜰 경우**

아래 명령어를 입력하여 충돌하는 패키지를 업데이트하세요.

```bash
pip install --upgrade intel-openmp mkl
```

**Q2. `WinError 127` 또는 `DLL load failed`가 뜰 경우**

0번 단계(Visual C++ 설치)를 안 했거나 **재부팅**을 안 한 경우입니다. 설치 후 컴퓨터를 껐다 켜주세요.

**Q3. `ModuleNotFoundError: No module named ...`**

가이드 외의 패키지가 없다면 담당자에게 문의하거나 `pip install [필요한 패키지명]`으로 설치하세요.

📧 담당자 이메일 : rkqls8522@naver.com
