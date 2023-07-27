LOG_MEMO=$1
LOGFILE_DIR="/disk3/dyf2/results/dsg_pami/logs/"
LOG_FILENAME=$(date "+%Y-%m-%d-%H-%M-%S")

# exec
log_filepath=$LOGFILE_DIR$LOG_FILENAME"-"$LOG_MEMO".log"
echo log file is saved as $log_filepath

python main.py --conf_path ./imagenet_resnet.hocon --id 01 --arc 18 --mn_perc 10 2>&1 | tee $log_filepath
