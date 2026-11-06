# 한강에서 MLPerf Training Llama 3.1 8B 재현하기 (빠른 안내)

자세한 설명은 [README](../README.md)(영문), 결과는 [RESULTS](RESULTS.md),
한강 특유의 문제와 해결책은 [TROUBLESHOOTING](TROUBLESHOOTING.md)에 있습니다.
모든 명령은 `/scratch` 아래 디렉터리에서 실행하세요. 한강은 `/scratch`나 `/appsdata`에서만 작업 제출을 허용합니다.

**어느 단계에서든 `./scripts/check_setup.sh`를 실행하면 끝난 단계와 다음에 실행할 명령을 알려 줍니다.** 단계마다 마지막 줄로 넣어 두었습니다.

```
  [ OK ] 0 sources: ...
  [ OK ] 1 data: /scratch/<사용자>/mlperf-llama31/data/llama3_1_8b_preprocessed_c4_dataset
  [ .. ] 2 base image: import job running (8.9G downloaded)
  [ .. ] 3 aws-ofi-nccl: needs the base image first
  ...
next: wait for the kv6-import job (squeue -u $USER)
```

`[ OK ]`는 끝남, `[ .. ]`는 진행 중이거나 아직 안 함, `[FAIL]`은 문제가 있음을 뜻합니다.

## 준비물과 소요 시간

| 항목 | 내용 |
|---|---|
| 계정 | GH200 파티션에 작업을 낼 수 있어야 함. **그룹마다 쓸 수 있는 파티션이 다름** (예: 어떤 계정은 `5thEx`만 가능). 네 파티션(`normal`, `natStr`, `5thEx`, `intTest`)의 노드는 모두 같은 GH200 |
| `--comment` | `showappl`로 확인. 기본값 `field=efficientai;appl=megatron-lm` 사용 가능 |
| 용량 | 약 150 GB (데이터 84 GB, tokenizer 다운로드 약 30 GB, 컨테이너 약 20 GB) |

새 계정으로 처음부터 해 본 소요 시간 (2026-11-06):

| 단계 | 시간 |
|---|---|
| 데이터 다운로드 (84 GB) | 빠를 때 약 2시간(분당 약 0.8 GB), 느릴 때는 훨씬 더. 다른 `/scratch`에서 복사하면 약 2분 |
| base 이미지 가져오기 (배치 작업) | 2~60분 (NGC 다운로드 속도에 따라) |
| aws-ofi-nccl 빌드 | 약 5분 |
| 벤치마크 컨테이너 빌드 | 11~15분 (예전 빌드는 70분까지) |
| 2노드 NCCL 점검 | 약 5분 |
| 4노드 시험 | 6~10분 |

## 0. 저장소 받기와 파티션 확인

```bash
cd /scratch/$USER
git clone https://github.com/hwang2006/mlperf-training-llama31-8b-on-supercomputer.git
cd mlperf-training-llama31-8b-on-supercomputer
source env.sh
for P in normal natStr 5thEx intTest; do printf "%-8s " $P; sbatch --test-only -p $P -N 1 --gres=gpu:4 -t 10 --comment="$SLURM_COMMENT" --wrap=hostname 2>&1 | tail -1; done
```

`Job ... to start at ...`이 나오는 파티션을 씁니다(`allocation failure`는 권한 없음). 아래 명령의 `5thEx`를 그 파티션으로 바꿔 **개인 설정 파일 `env.local.sh`**에 적습니다. 이 파일에는 `SLURM_COMMENT`, `DATA_ROOT`, `CONT_DIR`도 적을 수 있고, git이 무시하므로 저장소를 갱신해도 남습니다.

```bash
echo 'export SLURM_PARTITION=5thEx' >> env.local.sh      # env.local.sh는 git이 무시하는 개인 설정 파일
source env.sh && echo "$SLURM_PARTITION | $SLURM_COMMENT | $DATA_ROOT | $CONT_DIR"
./scripts/check_setup.sh                   # 첫 줄에 partition=<내 파티션>
```

## 1. 소스와 데이터 (로그인 노드)

```bash
./scripts/00_fetch_sources.sh | tail -2
setsid nohup ./scripts/04_prepare_data.sh > results/data-download.log 2>&1 < /dev/null &
./scripts/check_setup.sh                   # 1 data: download running (...G so far)
```

`setsid nohup ... &`로 띄우면 셸에 `[1]+ Done`이 바로 뜨지만, 실제로는 뒤에서 계속 받고 있습니다(`ps -u $USER | grep -E "prepare_data|wget"`). 끝나면 로그 마지막에 `DATA_READY`가 나옵니다.

같은 한강에 이미 데이터를 받은 사용자가 있고 읽기 권한을 주면, 복사가 훨씬 빠릅니다.

```bash
S=/scratch/<다른 사용자>/mlperf-llama31
mkdir -p "$PREPROC_DATA" "$TOKENIZER"
cp -a $S/data/llama3_1_8b_preprocessed_c4_dataset/. "$PREPROC_DATA/"
cp -a $S/tokenizer/llama3_1_8b_tokenizer_only/. "$TOKENIZER/"
./scripts/04_prepare_data.sh --verify      # DATA_READY 확인
./scripts/check_setup.sh                   # 1 data: [ OK ]
```

## 2. 컨테이너

**2a. base 이미지 가져오기** (GPU 노드에서 도는 배치 작업)

```bash
sbatch -p "$SLURM_PARTITION" --comment="$SLURM_COMMENT" scripts/01_import_base_image.sbatch
./scripts/check_setup.sh                   # 2 base image: import job running → 끝나면 [ OK ]
```

`2 base image`가 `[ OK ]`가 될 때까지 기다린 뒤 2b로 갑니다. 작업이 `squeue`에서 사라지고, 로그 `results/kv6-import-<작업번호>.log` 끝이 `CHECK_EXIT=0`이면 끝난 것입니다. 로그는 대부분의 시간 동안 `Downloading 78 missing layers...`에 머물러 있으니, 진행 상황은 `du -sh $ENROOT_CACHE_PATH`로 보세요. 시간 제한에 걸리면 같은 명령으로 다시 제출하면 됩니다. 이미 받은 레이어는 다시 받지 않습니다.

**2b. 2a가 끝난 뒤에** 플러그인과 컨테이너 빌드

```bash
./scripts/03_build_ofi_plugin.sh 2>&1 | tee results/ofi-build.log | tail -3      # OFI_BUILD_OK
setsid nohup ./scripts/02_build_container.sh > results/build.log 2>&1 < /dev/null &
./scripts/check_setup.sh                   # 3 [ OK ], 4 container: build running → 끝나면 [ OK ]
```

**두 줄을 모두 실행하세요.** 첫 줄은 `OFI_BUILD_OK`를 출력하고 끝나고, 둘째 줄이 컨테이너 빌드를 뒤에서 시작합니다(`./scripts/check_setup.sh`에 `4 container: build running`). 빌드 로그 끝에 `BUILD_EXIT=0`이 나오고 `$CONT_DIR`에 `kv6-nv60-llama31-8b-arm64.sqsh`(약 10 GB)가 생기면 끝입니다. 2a가 끝나기 전에 실행하면 `Base image not found`로 바로 멈춥니다.

## 3. 점검

```bash
sbatch -N 2 -p "$SLURM_PARTITION" --comment="$SLURM_COMMENT" --exclude="$EXCLUDE_NODES" slurm/nccl_test.sbatch
grep -E "^##########|world=|allreduce +(512|2048) MB" $(ls -t results/kv6-nccl-*.log | head -1)
./submit.sh 4 MAX_STEPS=200 TIME=00:40:00
./scripts/check_setup.sh                   # 모두 [ OK ]이면 끝에 "setup complete"
```

| 점검 | 기대값 |
|---|---|
| NCCL C1~C3 (Slingshot) | 2 GB에서 버스 대역폭 약 297 GB/s |
| NCCL C4, A | 약 10~12 GB/s (TCP. 정상) |
| 4노드 시험 | 첫 loss 약 12.58, 학습 처리량 약 24 samples/s |

4노드 시험에서 한 랭크만 `CUDA out of memory`로 실패하면, 그 노드의 GPU에 다른 것이 메모리를 잡고 있던 경우입니다(node1802에서 한 번 발생, 몇 시간 뒤에는 정상). 다시 내거나, 그 노드를 빼고 냅니다: `EXTRA_EXCLUDE=node1802 ./submit.sh 4 MAX_STEPS=200 TIME=00:40:00`

## 4. 측정

```bash
./submit.sh 4                                       # 약 140분
./submit.sh 2                                       # 약 270분
./submit.sh 1 TP=2 TIME=12:00:00                    # 약 11.5시간
./submit.sh 128 GBS=128 TP=1 CP=4 LR=0.001 WARMUP_STEPS=256 BUCKET_SIZE=2000000000 MAX_STEPS=1200000 TIME=00:40:00 NETPROFILE=cscs   # 최고 기록 구성, 약 14분
setsid nohup ./scripts/submit_queue.sh recipes/best_per_scale.txt > results/queue.log 2>&1 < /dev/null &
python3 tools/ttq_status.py --summary results/runs/kv6-l8b-*.out
python3 tools/collect_runs.py results/runs --sacct     # 모든 실행의 설정·결과·곡선을 results/raw/에 정리
```

스케일별 명령과 GBS별 하이퍼파라미터는 [recipes/README.md](../recipes/README.md)에 있습니다.

## 결과 요약

**재현 확인**: 새 계정(q667a01, `5thEx`)에서 저장소를 처음부터 따라 해 4노드 TTT 131.8분(원래 140.6분, 학습 처리량 0.1% 차이)으로 재현됐습니다.

128노드(GH200 512개), GBS 128에서 **13.77분**이 최고 기록입니다(MLPerf 평가 방식). 결과는 두 트랙으로 나눕니다. **트랙 1**은 MLPerf처럼 12,288 sample마다 평가하고 GBS 128 이하만 쓰며, **트랙 2**는 종료 조건만 같고 GBS와 평가 일정이 자유입니다. 노드를 늘리려면 TP·CP를 노드 안에 두기 위해 GBS를 키워야 하는데, GBS 128을 넘으면 목표까지 필요한 sample이 거의 batch만큼 늘어나 128노드가 최적점이었습니다. 그래프와 수치는 [RESULTS](RESULTS.md)에 있습니다.

## 꼭 알아 둘 점

| 항목 | 내용 |
|---|---|
| 사용자당 제출 한도 | 5개. `scripts/submit_queue.sh`가 빈자리가 날 때마다 다음 작업을 제출 |
| 자동 감시 | `submit.sh`가 작업마다 감시 스크립트를 띄워, loss가 NaN이거나 로그가 5분 이상 멈추면 자동 취소 |
| 네트워크 설정 | 기본값 `NETPROFILE=cscs-nordzv`. 기본 Slingshot 설정에서는 학습 중 통신이 멈추는 문제가 있었음 |
| 제외 노드 | `env.sh`의 `EXCLUDE_NODES`. 첫 step부터 NaN을 일으킨 노드(가장 유력: node0320~0324) 등 28개 |
| 1노드 | GPU 메모리(96 GB) 때문에 `TP=2` 필요 |
| 문제 노드 한 번만 빼기 | `EXTRA_EXCLUDE=nodeA,nodeB ./submit.sh ...` (`env.sh`의 목록에 덧붙음) |
| 큰 GBS | GBS 128 이상이면 `MAX_STEPS`가 자동으로 줄어듦 (GBS 256에서 초기화 시간 초과 방지) |
| 규모별 병렬화 | 128노드까지는 CP 위주(TP1·CP4 등), 256노드부터는 TP4와 큰 gradient 묶음(`BUCKET_SIZE=2000000000`) |
| 평가 일정 (비공식) | `EVAL_START=<sample> EVAL_EVERY=<sample>`로 첫 평가를 늦추고 간격을 좁힐 수 있음. MLPerf 규정과는 다름 |
| 결과의 성격 | 모두 1회 실행이며 MLCommons 공식 심사를 거치지 않은 결과. 공식 점수는 10회 실행 평균 |
