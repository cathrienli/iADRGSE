# iADRGSE
 **Source code for "iADRGSE : A graph embedding and self-attention encoding for identifying adverse drug reaction"**
![Image text](https://github.com/cathrienli/iADRGSE/blob/main/iADRGSE.png)
**&(a) Graph channel. We perform the RDKit tool to convert the drug SMILES into chemical structure graphs and feed them into a pretrained GIN network to learn graph-based structural information. (b) Sequence channel. The preprocessing unit utilizes Open Babel software to generate molecular substructure sequences from the SMILES of drugs. The substructure sequences are represented as a one-dimensional sequence vectors through the embedding layer. The encoder unit implements a multi-head self-attention mechanism to further extract the correlation information of each substructure. The feed-forward unit is a multi-fully connected layer, which receives encoded information of the upper layer to obtain the final sequence-based structural information of drugs. (c) Prediction module. These two types of structural information are concatenated and then mapped to the size of the labels through an affine transformation for multi-label prediction.**
# Required packages
**Python == 3.7**  
**PyTorch == 1.6**
# Dataset
**label(2248).csv   (Drug Label Information)**  
**smiles(2248).smlies    (Drug smiles)**
# Usage
  **train.ipynb**
