# Dataset Format Specification

## Overview
The GIL-DDI model requires three types of knowledge graphs and DDI interaction data. All data should be derived from DrugBank database.

## Required Files

### 1. Knowledge Graph Files

#### dataset1.txt - Drug-Chemical Entity Relationships
Format: `DrugName//ChemicalEntity//RelationType`

Example:
```
Aspirin//Salicylic acid//metabolite
Aspirin//COX-1//target
Ibuprofen//Propionic acid//metabolite
```

#### dataset2.txt - Drug Substructure Relationships  
Format: `DrugName//Substructure//RelationType`

Example:
```
Aspirin//Benzene ring//contains
Aspirin//Carboxyl group//contains
Ibuprofen//Isobutyl group//contains
```

#### dataset3.txt - Drug-Drug Interaction Relationships
Format: `DrugName1//DrugName2//InteractionType`

Example:
```
Aspirin//Warfarin//increases_effect
Aspirin//Ibuprofen//additive_effect
```

### 2. Additional Data Files

#### drug_smiles.csv
SMILES representations of drugs

Columns:
- `drug_name`: Name of the drug
- `smiles`: SMILES string representation

#### drug_maccs.csv
MACCS keys for drugs (166-bit fingerprint)

Columns:
- `drug_name`: Name of the drug
- `maccs_key_1` to `maccs_key_166`: Binary values for each MACCS key

#### ddi_interactions.csv
Complete DDI interaction events

Columns:
- `drug1`: First drug name
- `drug2`: Second drug name
- `mechanism`: Mechanism of interaction
- `action`: Action/effect of interaction
- `level`: Severity level (optional)

## Data Preparation Steps

1. **Obtain DrugBank License**: Visit https://www.drugbank.com and apply for an academic license

2. **Download Data**: Download the full database in XML or CSV format

3. **Extract Required Information**:
   - Drug names and IDs
   - Chemical structures (SMILES)
   - Drug targets and enzymes
   - Drug-drug interactions with mechanisms

4. **Process Data**:
   - Convert to the required format shown above
   - Generate MACCS keys from SMILES using RDKit
   - Create knowledge graph files

5. **Validate Data**:
   - Ensure all drug names are consistent across files
   - Check for missing values
   - Verify file formats

## Dataset Statistics

After preparation, verify your dataset has the following characteristics:

- **Number of drugs**: ~500-600 (depending on DrugBank version)
- **Number of DDI events**: ~65 unique mechanism-action pairs
- **Knowledge graph 1**: Drug-chemical entity relationships
- **Knowledge graph 2**: Drug substructure relationships  
- **Knowledge graph 3**: Drug-drug interaction relationships

## Notes

- Ensure all files use UTF-8 encoding
- Use `//` as the delimiter in .txt files
- Drug names must be consistent across all files
- Missing values should be handled appropriately during preprocessing

## Citation

Remember to cite DrugBank in your work:

```
Knox C, Wilson M, Klinger CM, et al. DrugBank 6.0: the DrugBank Knowledgebase for 2024. 
Nucleic Acids Res. 2024 Jan 5;52(D1): D1265-D1275. doi: 10.1093/nar/gkad976.
```



