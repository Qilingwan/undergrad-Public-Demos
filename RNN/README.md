# RNN for Name Generation

## 1. Project Description

(a) This project implements a recurrent neural network (RNN) to generate plausible names based on a dataset of existing names<br>
(b) The model uses character-level sequences to learn patterns in name structures<br>
(c) It demonstrates training and sampling techniques for text generation tasks<br>

## 2. Tech Stack / Tools Used

(a) Python 3.11+<br>
(b) PyTorch<br>
(c) NumPy<br>
(d) Matplotlib<br>

## 3. Objectives / Tasks

(a) Load and preprocess name data from text file<br>
(b) Create character alphabet and encoding mappings<br>
(c) Define dataset and dataloader for sequence processing<br>
(d) Build RNN model for character prediction<br>
(e) Train model on name sequences<br>
(f) Implement generation functions for new names with optional prefixes and constraints<br>

## 4. Implementation / Methods

### 4.1 Import Libraries

(a) Imported standard Python modules: os, random, math, itertools, string, time, json<br>
(b) Imported numpy for array operations<br>
(c) Imported torch and submodules for neural network implementation<br>
(d) Imported matplotlib.pyplot for visualization<br>

### 4.2 Set Hyper Parameters

(a) Defined constants: DATA_PATH for input file, EPOCHS=15, BATCH_SIZE=256, EMB_SIZE=32, HIDDEN_SIZE=128, LR=3e-3<br>
(b) Set DEVICE to CUDA if available, otherwise CPU<br>
(c) Set random seeds for reproducibility across random, numpy, and torch<br>

### 4.3 Load Data

(a) Read raw names into rawNames list from file, cleaned with cleanName to title case<br>
(b) Extracted unique characters to form alphabet, added SOS '^' and EOS '$' tokens<br>
(c) Created char2idx and idx2char mappings, set VOCAB_SIZE<br>

### 4.4 Set Database

(a) Defined NameDataset class inheriting from Dataset<br>
(b) Implemented encode method to convert names to index sequences with SOS/EOS<br>
(c) __getitem__ returned input sequence (x) and target sequence (y shifted by 1) as idxSeq<br>
(d) Defined collate function for padding batches with EOS token using padValue, paddedX, paddedY<br>
(e) Created trainLoader with shuffling and custom collate<br>

### 4.5 Construct Model

(a) Defined CharRNN class inheriting from nn.Module with parameters vocabSize, embSize, hiddenSize<br>
(b) Included Embedding layer (vocabSize to embSize)<br>
(c) GRU layer (embSize to hiddenSize, batch_first=True)<br>
(d) Linear layer (hiddenSize to vocabSize) for logits<br>
(e) Forward pass: embed input, pass through GRU, apply linear to get logits<br>
(f) Instantiated model on DEVICE, CrossEntropyLoss ignoring EOS, Adam optimizer<br>

### 4.6 Training

(a) Defined trainEpoch function to compute loss over batches using trainLoader<br>
(b) Model in train mode, zero grad, forward pass, loss calculation, backward, step<br>
(c) Accumulated loss in totalLoss weighted by batch size<br>
(d) Looped over EPOCHS calling trainEpoch and printing average loss<br>

### 4.7 Construct Generation Function

(a) Defined sampleName for generating names with prefix, maxLen, temp, returnLogprobs<br>
(b) Encoded prefix into inputIdxs, fed to model, sampled subsequent chars using multinomial<br>
(c) Collected logprobs in probsLog and prefix if requested<br>
(d) Defined visualizeGen to plot probability distributions per character<br>
(e) Defined genWithConstraints to filter samples matching suffix using nTries attempts<br>
(f) In __main__: demonstrated sampling with demoPrefix, constraint generation, visualization<br>

## 5. Results / Outputs

<p align="center">
  <img src="outputDemos/generatedName.png" alt="Generated Name">
</p>

<p align="center">
  <img src="outputDemos/favorateName.png" alt="Favorate Name">
</p>


## 6. Conclusion / Insights

(a) The RNN effectively captures character dependencies in names for generation<br>
(b) Temperature parameter controls diversity in sampling outputs<br>
(c) Constraint functions enable targeted name creation with prefix/suffix<br>
(d) Potential extensions include larger datasets or advanced architectures like LSTM<br>

## 7. Acknowledgements / References

(a) PyTorch Official Documentation: https://pytorch.org/docs/stable/index.html<br>

(b) Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *Empirical Methods in Natural Language Processing (EMNLP)*. (Introduced the Gated Recurrent Unit - GRU.)<br>