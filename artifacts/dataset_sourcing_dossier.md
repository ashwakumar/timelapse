# AeroGuard — Dataset Sourcing & Validation Dossier

## 1. Problem Formulation & Target User
- **Target User**: Aerospace Fleet Reliability Engineer & Maintenance Operations Manager
- **Organization**: Commercial & Defense Aviation Operators / MRO Facilities
- **Core Mission**: Proactively detect sub-component gas-turbine degradation, forecast Remaining Useful Life (RUL) with high precision, and schedule shop-level maintenance before unscheduled engine removals (UER) or in-flight shutdowns (IFSD).

## 2. Candidate Dataset Assessment
| Dataset ID | Candidate Name | Regimes | Fault Modes | Sensors | Score | Selected |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `nasa/cmapss-fd001` | NASA C-MAPSS Turbofan Degradation (FD001) | 1 | 1 | 21 | 9.5/10 | ✅ Yes |
| `nasa/cmapss-fd002` | NASA C-MAPSS Turbofan Degradation (FD002) | 6 | 1 | 21 | 8.0/10 | ❌ No |
| `nasa/ncmapss` | NASA N-CMAPSS Commercial Aircraft Telemetry | 9 | 2 | 47 | 7.8/10 | ❌ No |
| `phm08-challenge` | PHM08 Prognostics Data Challenge | 6 | 1 | 21 | 7.2/10 | ❌ No |

### Selection Justification
Optimal baseline benchmark: single operating regime eliminates confounding ambient temperature/altitude effects, isolating pure thermodynamic degradation dynamics. Extensively benchmarked across academic literature with established scoring functions.

## 3. Verified Physical Data Integrity
- **Raw File**: `data/raw/train_FD001.txt`
- **SHA-256 Checksum**: `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`
- **Total Operational Rows**: 20,631
- **Engine Population**: 100 unique turbofans
- **Engine Lifetimes**: Min = 128 cycles, Max = 362 cycles, Mean = 206.31 cycles
- **Active Diagnostic Sensors**: 14 selected thermodynamic channels
