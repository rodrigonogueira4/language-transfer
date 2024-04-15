# Log into the machine

rm -rf ~/language-transfer;
git clone --branch exp23 https://github.com/rodrigonogueira4/language-transfer.git;
rm -rf ~/language-transfer/t5x;
cd ~/language-transfer;
git clone --branch main https://<code-here>@github.com/maritaca-ai/t5x.git;
cd ~/language-transfer/t5x;
python3 -m pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html;
python3 -m pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html;
pip install -U pyglove==0.4.3


cd ~/language-transfer;nohup bash scripts/run_all_pretrainings.sh small 6B ar