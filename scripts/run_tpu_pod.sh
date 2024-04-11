
TPU_NAME=joao-v4-64-on-demand
ZONE=us-central2-b
PROJECT=robotic-tiger-387019

gcloud compute tpus tpu-vm delete $TPU_NAME   --zone=$ZONE  --project=$PROJECT

trial=1

while ! gcloud compute tpus tpu-vm create $TPU_NAME   --zone=$ZONE  --project=$PROJECT --accelerator-type=v4-64   --version=tpu-vm-v4-base  --preemptible
do
  echo Trial number: $trial
  sleep 3
  ((trial+=1))
done


gcloud alpha compute tpus tpu-vm ssh $TPU_NAME \
  --zone $ZONE \
  --project=$PROJECT \
  --worker=all \
  --command="pip install 'jax[tpu]==0.4.6' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; rm -rf ~/language-transfer; git clone --recurse-submodules --branch backup-scripts https://github.com/rodrigonogueira4/language-transfer.git;cd ~/t5x; python3 -m pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html;pip install -U pyglove==0.4.3"


gcloud alpha compute tpus tpu-vm ssh $TPU_NAME \
--zone $ZONE \
--project=$PROJECT \
--worker=all \
--command="cd ~/language-transfer;nohup bash ./run_all_pretrainings.sh 60M ar 550M > output.log 2>&1 &"


gcloud alpha compute tpus tpu-vm ssh $TPU_NAME \
--zone $ZONE \
--project=$PROJECT \
--worker=0 \
--command="tail -1000f ~/language-transfer/output.log"
