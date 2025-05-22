# Instruction-Tuning LLaMA for Synthetic Medical Note Generation: Bridging Data Privacy and Utility in Downstream Tasks

This repository contains the code belonging to the paper "Instruction-Tuning LLaMA for Diverse Synthetic Medical Note Generation". The purpose of this study was to generate synthetic Swedish and English discharge summaries that preserve privacy while mimicing the task-relevant properties of the real data needed to build high-performing 
downstream systems. The code in this repository is structured in different chapters similar to the paper. We recommend reading the 
corresponding chapters alongside investigating the code to better understand the purpose and desired outcome of each single part of the code. Note, that you need to get access
to the datasets and several language models from external sources (always explained in the concerning chapters). Keep in mind that this might take a while and requires planning ahead.
Last, don't forget that you are working with sensitive medical data that requires special care. For example, you are not 
allowed to upload any of this data to online APIs, meaning that all models used in this study must be downloaded and saved locally on your device/server before using.

## Acknowledgements

The code in this repository builds on several other studies and includes code from other repositories. Specifically it includes:
1. The [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) project in `generation/axolotl` for the fine-tuning process. The code of the original repository was not changed 
but only complemented with some additionals files closer specified in `generation/axolotl/changes.md`.
2. The [medical-coding-reproducability](https://github.com/JoakimEdin/medical-coding-reproducibility/tree/main) repository in `utility/plm_icd/medical_coding/` for building medical coding models. 
Some slight modifications were made to the source code closer specified in `utility/medical_coding/plm_icd/changes.md`
3. The ROUGE-5 implementation of the [mdpi2021-textgen](https://github.com/nedap/mdpi2021-textgen/tree/main/textgen/evaluation) in `privacy/rouge_5`. 
The changes made to this code are closer specified in `privacy/rouge_5/changes.md`.


## Environments

There are four different docker containers you will need to run for different parts of this code. It is always specified which container to run for which part of the code and which folders to mount. Always make sure you're running the correct container! 
Pull the following two images from docker hub:
```sh 
docker pull winglian/axolotl:main-20241202
``` 
```sh 
docker pull vllm/vllm-openai:v0.6.4.post1
```
Build the remaining four images from the dockerfiles save in  the `docker` folder by running
```sh 
docker build -t medcode:edin -f docker/DockerfileMedcode .
```
```sh 
docker build -t general:latest -f docker/DockerfileGeneral .
```
Due to the sensitivity of the data, you might need to transfer these images to a server with no internet access. In this case, save it by running
```sh 
docker save -o /path/for/generated/tar/file image_name
```
transfer the `.tar` file to your server and load it by running
```sh 
docker load -i /path/to/image/tar/file
```
Now you're ready to run all docker containers needed in this study and start your experiments.

## 1. Data Preprocessing

Set the environment by running the medical coding docker image mounting the `data`, `utility/medical_coding/plm_icd`, and `preprocessing` folders:
```sh 
sudo docker run --gpus all -v utility/medical_coding/phi_ner:/medical_coding -v data:/data -v preprocessing:/preprocessing -it medcode:edin bash
```

### 1.1 MIMIC

1. Get access to the [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) dataset on PhysioNet. Note, that you need to complete a training that takes a couple of hours to get access.
2. Then download MIMIC-IV and MIMIC-IV-NOTE into the folders specified in `/data/mimic`. 
3. Change the `DOWNLOAD_DIRECTORY_MIMICIV` and `DOWNLOAD_DIRECTORY_MIMICIV_NOTE` in `medical_coding/src/settings.py` to your respective paths.
4. Run 
```sh 
python /medical_coding/prepare_data/prepare_mimiciv.py 
```
to obtain the new file `/medical_coding/files/data/mimiciv_icd10/mimiciv_icd10.feather` alongside the splits for the dataset.

5. Run
```sh 
python /preprocessing/mimic/preprocessing.py --input_file /medical_coding/files/data/mimiciv_icd10/mimiciv_icd10.feather --output_file /medical_coding/files/data/mimiciv_icd10/mimiciv_icd10.feather
```
to preprocess the medical notes (use an alternative output file if you want to keep the original dataframe).

6. Run
```sh 
python /preprocessing/mimic/transcriptions.py --icd_file /preprocessing/mimic/ICD10_descriptions_mimic.csv --mimic_file /medical_coding/files/data/mimiciv_icd10/mimiciv_icd10.feather --output_file /medical_coding/files/data/mimiciv_icd10/mimiciv_icd10.feather 
```
to obtain textual descriptions of the ICD-10 codes (use an alternative output file if you want to keep the original dataframe).

7. Run 
```sh 
python /preprocessing/mimic/json_splits.py --notes_file /medical_coding/files/data/mimiciv_icd10/mimiciv_icd10.feather --splits_file /medical_coding/files/data/mimiciv_icd10/mimiciv_icd10_split.feather
```
to obtain JSON files as required to create prompts in the ALPACA format for MIMIC-S, MIMIC-L, and MIMIC-E.

### 1.2 SEPR
1. Get access to SEPR II by contacting the [Health Bank](https://www.dsv.su.se/healthbank) at Stockholm University for access. Create a `.feather` file similar to `/medical_coding/files/data/mimiciv_icd10/mimiciv_icd10.feather` by substituting the `text`and `target`columns with the SEPR data and store as `/medical_coding/files/data/sepr/sepr_icd10.feather`
2. Run
```sh 
python /preprocessing/sepr/transcriptions_swed.py --icd_file /preprocessing/sepr/ICD10_descriptions_sepr.csv --sepr_file /medical_coding/files/data/sepr/sepr_icd10.feather --output_file /medical_coding/files/data/sepr/sepr_icd10.feather
```
to obtain textual descriptions of the ICD-10 codes (use an alternative output file if you want to keep the original dataframe).

3. Run 
```sh 
python /preprocessing/sepr/json_splits_swed.py --notes_file /medical_coding/files/data/sepr/sepr_icd10.feather --splits_file /medical_coding/files/data/sepr/sepr_icd10_split.feather 
```
to obtain JSON files as required to create prompts in the ALPACA format for SEPR-S, SEPR-L, and SEPR-E.

## 2. Synthetic Medical Note Generation

### 2.1 Fine-tuning

Download LLaMA-3.1-8B (or any other model you want to use) from Hugggingface by running 
```sh 
import os 
os.environ['HF_TOKEN'] = ''
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir-use-symlinks False --local-dir path/to/local/dir
```
and save the model locally in the `models` folder. 

Set the environment by running the Axolotl docker image mounting the `data`, `generation/axolotl`, and `models` folders:
```sh 
sudo docker run --gpus all -v data:/data -v generation/axolotl:/axolotl -v models:/models -it winglian/axolotl:main-20241202 bash
```

#### 2.1.1 MIMIC

1. Go to `/axolotl/configs/ft_llama_mimic.yaml` and make sure the `base_model` (here LLaMA-3.1-8B), `datasets/path` (here MIMIC-L), and `output_dir` are correct. 
2. From the `/axolotl` directory run
```sh 
CUDA_VISIBLE_DEVICES="" python -m axolotl.cli.preprocess configs/ft_llama_mimic.yaml
```
to preprocess the dataset before fine-tuning.

3. From the `/axolotl` directory run
```sh 
CUDA_VISIBLE_DEVICES="" accelerate launch -m configs/ft_llama_mimic.yaml --deepspeed deepspeed_configs/zero3.json
```
to start the fine-tuning process.

4. From the `/axolotl` directory run
```sh 
CUDA_VISIBLE_DEVICES="" python3 -m axolotl.cli.merge_lora configs/ft_llama_mimic.yaml
```
to merch the LORA adapters back to the base model. The model will be saved in the subdirectory `merged` in your output path storing the fine-tuned model.

#### 2.1.2 SEPR

1. Go to `/axolotl/configs/ft_llama_sepr.yaml` and make sure the `base_model` (here LLaMA-3.1-8B), `datasets/path` (here SEPR-L), and `output_dir` are correct. 
2. Go to `/axolotl/src/axolotl` and rename the file `prompters.py` to  `prompters_org.py` and the file `prompters_swed.py` to `prompters.py`. This has to be done to create Swedish prompts. Don't forget to undo this change once you're done fine-tuning on the Swedish data.
3. From the `/axolotl` directory run
```sh 
CUDA_VISIBLE_DEVICES="" python -m axolotl.cli.preprocess configs/ft_llama_sepr.yaml
```
to preprocess the dataset before fine-tuning.

4. From the `/axolotl` directory run
```sh 
CUDA_VISIBLE_DEVICES="" accelerate launch -m configs/ft_llama_sepr.yaml --deepspeed deepspeed_configs/zero3.json
```
to start the fine-tuning process.

5. From the `/axolotl` directory run
```sh 
CUDA_VISIBLE_DEVICES="" python3 -m axolotl.cli.merge_lora configs/ft_llama_sepr.yaml
```
to merch the LORA adapters back to the base model. The model will be saved in the subdirectory `merged` in your output path storing the fine-tuned model.

### 2.2 Inference

Set the environment by running the vLLM docker image mounting the `data`, `generation/axolotl`, and `generation/inference` folders:
```sh 
docker run --entrypoint bash --gpus all -v /data:/data -v /generation/axolotl:/axolotl -v /generation/inference:/inference -it vllm-openai:v0.6.4.post1
```
#### 2.2.1 MIMIC

Run
```sh 
python3 /inference/inference_vllm_mimic.py --base_model path/to/fine-tuned/merged/llama --test_data path/to/sampling/file.json --file_out path/to/output/file.json
```
specifiying the path to the fine-tuned and merged LLaMA model (e.g., `/axolotl/output/mimic_s_llama/merged`) to the JSON file that should be used to generate the prompts for sampling (e.g., `/data/mimic/mimic_s.json`) and the output JSON file that stores the generated synthetic notes (e.g., `/data/mimic/synth_mimic_s.json`)

#### 2.2.2 SEPR

Run
```sh 
python3 /inference/inference_vllm_sepr.py --base_model path/to/fine-tuned/merged/llama --test_data path/to/sampling/file.json --file_out path/to/output/file.json
```
specifiying the path to the fine-tuned and merged LLaMA model (e.g., `/axolotl/output/sepr_llama/merged`), to the JSON file that should be used to generate the prompts for sampling (e.g., `/data/mimic/sepr_s.json`), and the output JSON file that stores the generated synthetic notes (e.g., `/data/mimic/synth_sepr_s.json`)

## 3. Fidelity Evaluation

Set the environment by running 
```sh 
sudo docker run --gpus all -v fidelity:/fidelity -v data:/data -v -it general:latest
```

Run
```sh 
python /fidelity/statistical_comparison.py --real_file path/real/documents.json \
            --field_real outputfield \
            --synthetic_file path/real/documents.json \
            --field_synthetic outputfield 
```
specifying the paths to the JSON files containing real and synthetic documents (e.g., `data/mimic/mimic_s.json` and `synth_mimic_s.json`) and the field containing the documents to analyze (e.g, `output` and `output1`). This will print a comparison of key statistical features of both datasets.

## 4. Utility: Medical Coding

Set the environment by running the medical coding docker image mounting the `data` and `utility/medical_coding/plm_icd` folders:
```sh 
sudo docker run --gpus all -v utility/medical_coding/plm_icd:/medical_coding -v data:/data -it medcode:edin bash
```

### 4.1 MIMIC

1. Download [RoBERTa-base-PM-M3-Voc](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-hf.tar.gz) and save the unzipped files.
2. Set the `model_path` parameter in `medical_coding/configs/plm_icd.yaml` and `configs/text_transform/huggingface.yaml` to the path of the RoBERTa model folder.
3. Create a `.feather` file containing the real or synthetic notes in the `text` column, the number of words in `num_words`, the ICD-10 diagnosis codes in `icd10_diag`, the ICD-10 procedure codes in `icd10_proc`, the combined codes in the `target`, the number of target in `num_target` and the ids in the `_id` column. Using the `/medical_coding/files/data/mimiciv_icd10/mimiciv_icd10.feather` as base, filtering for the ids and just substituting the documents simplifies this process a lot.
4. Store this file in the same directory as the file containing the splits file with `_id` and `split` columns. The splits used in this work are stored in `/medical_coding/files/data/mimiciv_icd10/mimiciv_icd10_split.feather` for training on MIMIC-L and `/medical_coding/files/data/mimiciv_icd10/splits_val_train.feather` for training on MIMIC-S. Change the `/medical_coding/configs/data/mimiciv_icd10.yaml` by specifying `dir`, `data_filename` and `split_filename`.
5. To train the medical coding model run
 ```sh 
python main.py experiment=mimiciv_icd10/plm_icd gpu=x callbacks=no_wandb trainer.print_metrics=true
```
specifying the GPU you want to use. 

6. If you want to evaluate a trained model run
 ```sh 
python main.py experiment=mimiciv_icd10/plm_icd gpu=x load_model=path/to/model/checkpoints trainer.epochs=0 callbacks=no_wandb trainer.print_metrics=true
```
specifying the GPU you want to use and model folder containing the checkpoints of your medical coding model.

### 4.2 SEPR

1. Get access to the model checkpoints of SweDeClin-BERT by contacting the [Health Bank](https://www.dsv.su.se/healthbank) at Stockholm University and store them in a folder.
2. Set the `model_path` parameter in `medical_coding/configs/plm_icd.yaml` and `configs/text_transform/huggingface.yaml` to the path of the SweDeClin-BERT model folder.
3. Create a `.feather` file containing the real or synthetic notes in the `text` column, the number of words in `num_words`, the ICD-10 diagnosis codes in `icd10_diag`, the ICD-10 procedure codes in `icd10_proc`, the combined codes in the `target`, the number of target in `num_target` and the ids in the `_id` column. 
4. Store this file in the same directory as the file containing the splits file with `_id` and `split` columns. The splits used in this work are stored in `/medical_coding/files/data/sper/sepr_icd10_split.feather` for training on SEPR-L and `/medical_coding/files/data/sepr/swed_splits_test_train.feather` for training on SEPR-s. Change the `/medical_coding/condigs/data/mimiciv_icd10.yaml` by specifying `dir`, `data_filename` and `split_filename`.
5. To train the medical coding model run
 ```sh 
python main.py experiment=mimiciv_icd10/plm_icd gpu=x callbacks=no_wandb trainer.print_metrics=true
```
specifying the GPU you want to use. 

6. If you want to evaluate a trained model run
 ```sh 
python main.py experiment=mimiciv_icd10/plm_icd gpu=x load_model=path/to/model/checkpoints trainer.epochs=0 callbacks=no_wandb trainer.print_metrics=true
```
specifying the GPU you want to use and the model folder containing the checkpoints of your medical coding model.

### 4.3 Error Analysis
Set the environment by running
```sh 
sudo docker run --gpus all -v utility/medical_coding:/medical_coding -v data:/data -it general:latest
```

The error analysis is tailored to the MIMIC data and needs some prior adaptation if desired to apply to the SEPR models.

#### 4.3.1 Predictions

To analyze key features of the predictions of the medical coding model compared to the targets, run
 ```sh 
python  /medical_coding/error_analysis/predictions.py --file path/to/prediction.feather --threshold x 
```
specifying the path to the test prediction file obtained in the evaluation of your medical coding model, and the optimal threshold also obtained in the evaluation of your medical coding model.

To get the percentage of predicted code sequences that are present in the training targets or form a subset of a code sequence from the training data, run
 ```sh 
python  /medical_coding/error_analysis/code_match.py --pred_file path/to/prediction.feather \ 
         --data_file path/to/training/data.feather \
         --split_file path/to/split/file.feather \
         --threshold x 
```
specifying the prediction file and threshold obtained during training as well as the path to the data file and splits file used during training.

#### 4.3.2 F1 vs. Code Frequency and Document Length
1. Generate a CSV file storing the F1 score and its frequency in the training data for each code by running
 ```sh 
python /medical_coding/error_analysis/get_f1_frequ.py --train_path path/to/training/data.feather \
                --split_path path/to/split/file.feather \
                --pred_file path/to/prediction.feather \
                --threshold x \
                --output_file path/to/output.csv
```
specifying the prediction file and threshold obtained during training as well as the path to the data file and splits file used during training and the path and name of the output file. For comparison, repeat this process for the model trained on real data and the model trained on synthetic data.

2. To plot F1 against code frequency and obtain spearman and pearson correlation coefficients, run
```sh
python /medical_coding/error_analysis/plot_f1_vs_frequ.py \
                --real_path path/to/f1_freq_df_real.csv \
                --synth_path path/to/f1_freq_df_synth.csv \
                --plot_output path/to/plot.png 
```
specifying the paths to the dataframes of the real-data and synthetic-data models generated in step 1 and the name of the plot file.

3. Generate a CSV file storing the F1 score and its length for each document in the test file running
 ```sh 
python /medical_coding/error_analysis/get_f1_doc_length.py --train_path path/to/training/data.feather \
                       --split_path path/to/split/file.feather \
                       --pred_file path/to/prediction.feather \
                       --threshold x \
                       --output_file path/to/output.csv
```
specifying the prediction file and threshold obtained during training as well as the path to the data file and splits file used during training and the path and name of the output file. For comparison, repeat this process for the model trained on real data and the model trained on synthetic data.

4. To plot F1 against document length and obtain spearman and pearson correlation coefficients, run
```sh
python /medical_coding/error_analysis/plot_f1_vs_length.py \
                        --real_path path/to/f1_freq_df_real.csv \
                        --synth_path path/to/f1_freq_df_synth.csv \
                        --plot_output path/to/plot.png 
```
specifying the paths to the dataframes of the real-data and synthetic-data models generated in step 3 and the name of the plot file.

#### 4.3.3 WF and OOF Errors

To analyze how many incorrect predictions are WF and OOF errors and plot the ICD-10 chapter distribution of the predictions in a pie chart, run
```sh
python /medical_coding/error_analysis/wf_oof_chapters.py \
    --preds path/to/predictions.feather \
    --threshold x \
    --plot_output path/to/plot.png 
```
specifying the prediction file and threshold obtained during training and the name of the plot file.

#### 4.3.4 Noise
To investigate whether noise contained in the synthetic data is widespread (H1) or rather concentrated in a subset of documents (H2) follow these steps:
1. Insert widespread noise in the real dataset by running 
```sh
python /medical_coding/error_analysis/noise_h1.py --file_path path/to/real.feather \
                    --split_path path/to/splits.feather \
                    --noise_percentage x \
                    --output_file_path path/to/output_file.feather
```                    
specifying the path to the real data where the noise is to be inserted, the respective splits file, the percentage of noise that should be added and the path and name of the output file containing the widespread noise in the training data. Repeat this process with different numbers of noice percentage.

2. Insert concentrated noise in the real dataset by running 
```sh
python /medical_coding/error_analysis/noise_h2.py --file_path path/to/real.feather \
                    --split_path path/to/splits.feather \
                    --noise_percentage x \
                    --output_file_path path/to/output_file.feather
```                    
specifying the path to the real data where the noise is to be inserted, the respective splits file, the percentage of noise that should be added and the path and name of the output file containing the widespread noise in the training data. Repeat this process with different numbers of noice percentage.

3. Use the files obtained in step 1 and 2 to train medical coding models as described in 4.1. Then plot F1 agains frequency and document length comparing the models trained on noisy real data to a model trained on the same amount of synthetic data and compare the curves. Analyze which hypothesis seems more likely by aligning more closely to the curve of the synthetic-data model and which percentage of noise achieves the most similar results. 

## 6. Privacy

Set the environment by running 
```sh 
sudo docker run --gpus all -v privacy:/privacy -v data:/data -it general:latest
```

### 6.1 MIMIC: ROUGE-5

1. To obtain a CSV file with ROUGE-5 scores between the training data and the synthetic data, run
```sh
python /privacy/rouge_5/rouge_similarity_ranking.py --realpath /path/to/training.json \
                    --real_field outputfield
                    --synthpath /path/to/synthetic.json \
                    --synth_field outputfield
                    --outdir path/to/output/dir \
                    --n_jobs x \
                    --batch_size x
```
specifying the path to both JSON files and the name of the fields containing the documents and the number of synthetic documents processed in parallel (`--n_job`) as well as the number of synthetic documents to process before saving (`--batch_size`)

2. To get the average, minimum, and maximum ROUGE-5 recall score of all documents and the 122 most similar documents, run
```sh
python /privacy/rouge_5/evaluate_rouge.py --rouge_file path/to/rouge.csv \
```
specifying the path to the CSV file storing the ROUGE-5 scores obtained in step 1.

3. For further evaluation of the 20 most similar document pairs, run
```sh
python /privacy/rouge_5/longest_sequence.py --rouge_file path/to/rouge.csv \
```
specifiying the path to the ROUGE CSV file to obtain statistics about the longest overlapping sequences, and
```sh
python /privacy/rouge_5/longest_sequence.py --train_file path/to/training.feather
--splits_file path/to/splits.feather
--rouge_file path/to/rouge.csv \
```
specifying the path to the training and split feather files and the ROUGE CSV file to obtain the overall count of overlapping 5-grams in the 20 most similar documents in the training data.

### 6.2 SEPR: 8-gram Overlap

To calculate the 8-gram overlap between the training data used to finetune LLaMA and the generated synthetic data, run
```sh
python3 /privacy/8_gram/ngram_overlap.py --original_file path/to/training.json  \
                    --original_field outputfield  \
                    --synthetic_file path/to/synthetic.json  \
                    --synthetic_field outputfield  \
```
specifying the path to both JSON files and the name of the fields containing the documents.

## 7. Readability and Medical Coherence

### 7.1 Questionnaire

The XML files of the questionnaires of the MIMIC and SEPR study can be found in `readability_coherence/questionnaire`. You can import them on [Soscisurvey](www.soscisurvey.de) to investigate and edit the study.

### 7.2 Evaluation

You can insert the results of the study into `readability_coherence/evaluation/study_evaluation.ipynb` to generate boxplots for the evaluation of the ratings, test for statistical significance and investigate the correlation of readability and medical coherence.
