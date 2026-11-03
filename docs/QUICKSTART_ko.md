# 한강에서 MLPerf Training Llama 3.1 8B 재현하기 (빠른 안내)

자세한 설명은 [README](../README.md)(영문), 결과는 [RESULTS](RESULTS.md),
한강 특유의 문제와 해결책은 [TROUBLESHOOTING](TROUBLESHOOTING.md)에 있습니다.
모든 명령은 `/scratch` 아래 디렉터리에서 실행하세요. 한강은 `/scratch`나 `/appsdata`에서만 작업 제출을 허용합니다.

## 1. 준비 (로그인 노드)

```bash
cd /scratch/$USER
git clone https://github.com/hwang2006/mlperf-training-llama31-8b-on-supercomputer.git
cd mlperf-training-llama31-8b-on-supercomputer
vi env.sh            # SLURM_COMMENT(필수, showappl 참고), DATA_ROOT, CONT_DIR 확인
source env.sh
./scripts/00_fetch_sources.sh
setsid nohup ./scripts/04_prepare_data.sh > results/data-download.log 2>&1 < /dev/null &   # 1시간 이상, 끝에 DATA_READY
```

## 2. 컨테이너

```bash
sbatch --comment="$SLURM_COMMENT" scripts/01_import_base_image.sbatch                    # 약 15분
./scripts/03_build_ofi_plugin.sh 2>&1 | tee results/ofi-build.log                       # 약 5분, OFI_BUILD_OK
setsid nohup ./scripts/02_build_container.sh > results/build.log 2>&1 < /dev/null &    # 40~70분, BUILD_EXIT=0
```

## 3. 점검

```bash
sbatch -N 2 -p normal --comment="$SLURM_COMMENT" slurm/nccl_test.sbatch    # 버스 대역폭 약 300 GB/s 기대
./submit.sh 4 MAX_STEPS=200 TIME=00:40:00                                   # 첫 loss 약 12.58, 약 24 samples/s
```

## 4. 측정

```bash
./submit.sh 64 GBS=64 CP=4 LR=0.0008 WARMUP_STEPS=64 TIME=00:50:00          # 최고 기록 구성, 약 19분
setsid nohup ./scripts/submit_queue.sh recipes/best_per_scale.txt > results/queue.log 2>&1 < /dev/null &
python3 tools/ttq_status.py --summary results/runs/kv6-l8b-*.out
```

스케일별 명령과 GBS별 하이퍼파라미터는 [recipes/README.md](../recipes/README.md)에 있습니다.

## 결과 요약

64노드(GH200 256개), GBS 64에서 **19.1분**이 최고 기록입니다. 노드를 늘리려면 GBS를 키워야 하는데, GBS 64를 넘으면 목표까지 필요한 sample이 처리량보다 빨리 늘어나 128노드 이상에서는 더 빨라지지 않았습니다. 그래프와 수치는 [RESULTS](RESULTS.md)에 있습니다.

## 꼭 알아 둘 점

| 항목 | 내용 |
|---|---|
| 사용자당 제출 한도 | 5개. `scripts/submit_queue.sh`가 빈자리가 날 때마다 다음 작업을 제출 |
| 자동 감시 | `submit.sh`가 작업마다 감시 스크립트를 띄워, loss가 NaN이거나 로그가 5분 이상 멈추면 자동 취소 |
| 네트워크 설정 | 기본값 `NETPROFILE=cscs-nordzv`. 기본 Slingshot 설정에서는 학습 중 통신이 멈추는 문제가 있었음 |
| 제외 노드 | `env.sh`의 `EXCLUDE_NODES`. 노드 간 CP 구성에서 NaN을 일으킨 노드 등 25개 (2026-11-02 기준) |
| 1노드 | GPU 메모리(96 GB) 때문에 `TP=2` 필요 |
| 큰 GBS | GBS 128 이상이면 `MAX_STEPS`가 자동으로 줄어듦 (GBS 256에서 초기화 시간 초과 방지) |
| 평가 일정 (비공식) | `EVAL_START=<sample> EVAL_EVERY=<sample>`로 첫 평가를 늦추고 간격을 좁힐 수 있음. MLPerf 규정과는 다름 |
| 결과의 성격 | 모두 1회 실행이며 MLCommons 공식 심사를 거치지 않은 결과. 공식 점수는 10회 실행 평균 |
