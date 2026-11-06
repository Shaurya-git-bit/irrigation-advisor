\# 💧 Punjab Irrigation Advisor



\### An AI advisor that tells a Punjab wheat farmer \*\*"IRRIGATE today"\*\* or \*\*"WAIT, the soil will still be wet in 3 days"\*\*, with proof for every answer.



\*\*Neural Storm 2026 · Track 2 Green Planet · Problem statement GP-02: AI-Based Precision Irrigation Advisory\*\*



\*\*Team:\*\* `<team name>` · \*\*Members:\*\* Shaurya Pandey

\*\*Live demo:\*\* `<Streamlit Cloud / Hugging Face link>`



\---



\## The problem



Punjab is draining its groundwater.



\- Extraction is \*\*156% of recharge\*\* (national average: 61%), CGWB 2025

\- \*\*111 of 153\*\* assessment units are over-exploited

\- \*\*95%\*\* of the state's water use is irrigation, pumped by about \*\*1.45 million tubewells\*\*



Most farmers still irrigate on a \*\*fixed calendar\*\*, even right before rain and even when the soil is still wet. Every unnecessary irrigation is groundwater that does not come back.



\## The solution



Give the farmer a daily decision they can trust: \*\*irrigate or wait\*\*, backed by a model that has been scored on years it never saw.



```

Farmer picks: district + date + safety level

&#x20;  → weather \& soil-moisture history (Open-Meteo ERA5 / ERA5-Land)

&#x20;  → 63 engineered features (rain memory, moisture trends, water balance, crop age)

&#x20;  → XGBoost: predicts soil moisture 3 days ahead

&#x20;  → Decision engine: expected moisture vs the district's dry limit + safety margin

&#x20;  → IRRIGATE / WAIT

&#x20;  → SHAP: why the model said it

&#x20;  → Water-saved counter for the season vs the fixed calendar

```



The farmer chooses how cautious to be (\*\*Save more water → Maximum safety\*\*). The app states the exact trade-off for each level, for example \*"skips about 65% of scheduled irrigations; misses about 16% of the ones crops really needed."\*



\---



\## Results (measured on unseen years, 23 districts)



| What we measured | Result |

|---|---|

| Soil-moisture error vs "tomorrow = today" at \*\*1 day\*\* | \*\*15.3% lower\*\* |

| ... at \*\*3 days\*\* | \*\*8.8% lower\*\* |

| ... at \*\*5 days\*\* | \*\*8.5% lower\*\* |

| Calendar irrigations the advisor safely skips (3-day advisor, test years 2024-25) | \*\*65.0%\*\* |

| Needed irrigations missed at that setting | \*\*15.6%\*\* (84% of real crop needs still covered) |

| Lead over the "check the field today" rule at the same crop safety | \*\*up to +6 points of water saved\*\* |

| Muktsar wheat season 2024-25 | \*\*7 of 8\*\* calendar irrigations skippable |



\*\*No leakage in the decision layer.\*\* The safety margin is tuned on \*\*2023 only\*\* and scored on \*\*2024-25\*\*. The advisor is compared with the rule a farmer already uses (look at today's soil), and each rule gets its own tuned margin, so the comparison is fair.



\### Same crop safety, more water saved

!\[Safety vs water saved](plots/plot1\_safety\_vs\_savings.png)

At every safety level, the AI advisor sits above the field-check rule. The gap is largest where safety matters most.



\### One real season: calendar vs advisor (Muktsar, 2024-25)

!\[Season timeline](plots/plot2\_season\_timeline.png)

Green ✖ marks days the fixed calendar would irrigate but the advisor says \*\*WAIT\*\*. Soil was already well above the dry limit, often right after rain.



\### Prediction vs reality, 3 days ahead

!\[Prediction vs actual](plots/plot3\_pred\_vs\_actual.png)

The model follows the seasonal moisture curve across wet spells and long dry-downs.



\### Better than "nothing changes" at every horizon

!\[Horizons](plots/plot4\_horizons.png)



\### Every answer comes with a reason

!\[SHAP waterfall](plots/shap\_waterfall.png)

SHAP shows which factors pushed the forecast up or down (moisture trend, recent rain, temperature, crop age). The app turns them into plain labels such as \*"Soil moisture today"\* and \*"1-day moisture change"\*.



\---



\## Why it works



\- \*\*Targets the change, not the level.\*\* The model predicts how much soil moisture will change over H days and adds it to today's value, which is a far harder baseline to beat than predicting the raw level.

\- \*\*Water-balance features.\*\* Rolling rain minus ET0 over 7/14/30 days, an antecedent precipitation index, days since rain, and multi-layer soil gradients.

\- \*\*Safety is a dial, not a guess.\*\* The margin is calibrated on held-out data, so the farmer picks a level and sees the real trade-off.

\- \*\*Honest benchmarks.\*\* Compared against persistence for accuracy and against the field-check rule for decisions.



\---



\## Repository structure



```

irrigation-advisor/

├── irrigation\_adviser.py        # Streamlit app: Advisor, Results and About pages

├── requirements.txt             # Python dependencies for the app

├── README.md                    # This file

├── model/

│   ├── xgb\_h3.json              # Trained XGBoost model (3-day-ahead soil moisture change)

│   ├── features.json            # Exact feature order the model expects

│   └── thresholds.json          # Per-district dry limits + calibrated safety margin

├── data/

│   └── app\_data.csv             # Pre-computed features for 2023-25 so the app needs no feature code

├── plots/

│   ├── plot1\_safety\_vs\_savings.png   # AI advisor vs field-check rule (water saved vs crop safety)

│   ├── plot2\_season\_timeline.png     # One wheat season: calendar vs advisor decisions

│   ├── plot3\_pred\_vs\_actual.png      # 3-day-ahead prediction vs actual soil moisture

│   ├── plot4\_horizons.png            # Error reduction vs baseline at 1/3/5 days

│   └── shap\_waterfall.png            # SHAP explanation for a single prediction

├── results/                     # Evaluation outputs behind the plots and app numbers

│   ├── horizon\_results.csv      # MAE of model vs persistence at each horizon

│   ├── fix1\_curves.csv          # Savings vs missed-irrigation curve for every safety margin

│   └── fix1\_final.csv           # Final 2024-25 scores at the margins chosen on 2023

└── notebook/

&#x20;   └── irrigation-ml.ipynb      # Full pipeline: data fetch, features, training, evaluation, export

```



\## Tech stack



Python · XGBoost · SHAP · pandas · NumPy · scikit-learn · Matplotlib · Streamlit · Open-Meteo Historical API (ERA5 / ERA5-Land)



\## Setup and run locally



```bash

git clone <repo-url>

cd irrigation-advisor

pip install -r requirements.txt

streamlit run irrigation\_adviser.py

```



Open the local URL Streamlit prints. Pick a \*\*district\*\*, a \*\*date\*\* and a \*\*safety level\*\*.



To reproduce the model, open `notebook/irrigation-ml.ipynb` (Kaggle, Internet ON for the first cell) and run all cells. It fetches the data, builds features, trains, evaluates and exports everything in `model/`, `data/`, `plots/` and `results/`.



\## What's nextsss



The app runs on the 2023-25 Open-Meteo archive. The next step is wiring in the live Open-Meteo rain forecast so the advisor can say "rain expected" as well as "soil will stay wet", then extending beyond wheat.



\## AI tools disclosure



Claude (Anthropic) was used for coding assistance, debugging and drafting this README. All modelling choices, evaluation design and results were run and verified by the team.



\## Team



\*\*Shaurya Pandey\*\*: data science and ML

