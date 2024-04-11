
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

# Install packages
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME \
  --zone $ZONE \
  --project=$PROJECT \
  --worker=all \
  --command="rm -rf ~/language-transfer;git clone --branch backup-scripts https://github.com/rodrigonogueira4/language-transfer.git; rm -rf ~/language-transfer/t5x; cd ~/language-transfer; git clone --branch main https://<code-here>@github.com/maritaca-ai/t5x.git;cd ~/language-transfer/t5x; python3 -m pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html;python3 -m pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; pip install -U pyglove==0.4.3"

# Run training script
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME \
--zone $ZONE \
--project=$PROJECT \
--worker=all \
--command="cd ~/language-transfer;nohup bash scripts/run_all_pretrainings.sh 550M 60M > output.log 2>&1 &"

# See logs
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME \
--zone $ZONE \
--project=$PROJECT \
--worker=0 \
--command="tail -1000f ~/language-transfer/output.log"
