

The hyperparameter search space is described in following Table. The learning rate is fixed at 0.001, while attention heads for the 
semantic module, weight decay, hidden dimensions for GNNs and dropout rate is evaluated with candidate values

<img width="540" height="259" alt="image" src="https://github.com/user-attachments/assets/ae526dc5-a8f2-42bf-ab95-bcc58dc6e674" />

We evaluate the BAAI general embedding (size: 33.4M params, BAAI/bge-small-en-v1.5) and Google Gemma (size: 300M, google/embeddinggemma-300m)
models to obtain an LLM-based encoding of the associated text

We also present the hyperparameter configurations for the obtained TAG-CSAN results in following Table. Rest of the configurations such as learning rate, weight decay,
training epochs are described in above Table.
<img width="736" height="283" alt="image" src="https://github.com/user-attachments/assets/0a7d916e-d96b-44cb-9f88-c768432b055a" />
