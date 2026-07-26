""" DATASET EXTRACTION """
EXCLUDE_DNI = True


""" DATA PROCESSING """
# Variables that are not normally distributed based on normality tests and manual inspection.
# Normality test used: scipy.stats.normaltest (2-sided chi squared probability for the hypothesis test).
CLEAR_NON_NORMAL_VARS = [
    'bun_last',
    'creatinine_last',
    'glucose_last',
    'glucose_mean_last24h',
    'inr_last',
    'pco2_last',
    'platelet_last',
    'po2_last',
    'ptt_last',
    'rdw_last',
    # 'urineoutput_24hr',
    # 'fluidbalance_24hr',
    'wbc_last',
    'platelet_last'
]

# Variables that are likely not normally distributed.
SEMI_NORMAL_VARS = [
    'temperature_mean_last_24h',
    'apsiii', 
    'potassium_last', 
    'spo2_mean_last24h',
    'hemoglobin_last',
    'admission_age',
    'ph_last',
    'hematocrit_last'
]
NON_NORMAL_VARS = CLEAR_NON_NORMAL_VARS + SEMI_NORMAL_VARS

NUMERICAL_CORRELATED = [
    'sbp_mean_last24h', # --> Correlated with MBP
    'dbp_mean_last24h', # --> Correlated with MBP
    'hematocrit_last', # --> Correlated with Hemoglobin
    'mcv_last', # --> Correlated with MCH
    'rbc_last', # --> Correlated with Hemoglobin
    'bun_last', # --> Correlated with Creatinine
    'chloride_last', # --> Correlated with Sodium
    'baseexcess_last', # --> Correlated with Bicarbonate
    'totalco2_last', # --> Correlated with PaCO2
    'glucose_last', # --> Correlated with average glucose over last 24h
    # 'pco2_last' # --> Correlated with bicarbonate
    'bicarbonate_last' # --> Correlated with pco2
]

# Tableone package formatting parameters:
FORMAT_PARAMS = {
    'limit': {
        'inhosp_mortality': 1,
        'gender': 1,
        'sepsis3': 1,
        'myocardial_infarct': 1,
        'congestive_heart_failure': 1,
        'peripheral_vascular_disease': 1,
        'cerebrovascular_disease': 1,
        'chronic_pulmonary_disease': 1,
        'liver_disease': 1,
        'renal_disease': 1,
        'malignant_cancer': 1,
        'diabetes': 1,
        'vp_last24h': 1,
        'vp_last12h': 1,
        'vp_last6h': 1,
    },
    # Levels must be quoted: tableone casts categorical levels to str before
    # matching. With ints ([1, 0]) the lookup fails, tableone warns "Order value
    # not found" and falls back to sorted order, so every binary variable is
    # reported at level 0 (characteristic ABSENT) under a label that reads as if
    # it were present.
    'order': {
        'inhosp_mortality': ['1', '0'],
        'gender': ['1', '0'],
        'sepsis3': ['1', '0'],
        'myocardial_infarct': ['1', '0'],
        'congestive_heart_failure': ['1', '0'],
        'peripheral_vascular_disease': ['1', '0'],
        'cerebrovascular_disease': ['1', '0'],
        'chronic_pulmonary_disease': ['1', '0'],
        'liver_disease': ['1', '0'],
        'renal_disease': ['1', '0'],
        'malignant_cancer': ['1', '0'],
        'diabetes': ['1', '0'],
        'vp_last24h': ['1', '0'],
        'vp_last12h': ['1', '0'],
        'vp_last6h': ['1', '0'],
    },
    'labels': {
        # Demographics
        'gender': 'Gender, Male',
        'admission_age': 'Admission age',
        'weight_admit': 'Weight',
        'race_cat': 'Ethnicity',

        # Comorbidities
        'sepsis3': 'Sepsis',
        'myocardial_infarct': 'Myocardial infarct',
        'congestive_heart_failure': 'Congestive heart failure',
        'peripheral_vascular_disease': 'Peripheral vascular disease',
        'cerebrovascular_disease': 'Cerebrovascular disease',
        'chronic_pulmonary_disease': 'COPD',
        'liver_disease': 'Liver disease',
        'renal_disease': 'Renal disease',
        'malignant_cancer': 'Malignant cancer',
        'diabetes': 'Diabetes',

        # Vasopressor usage
        # 'vp_last24h': 'Documented vasopressor use (last 24h)',
        # 'vp_last12h': 'Documented vasopressor use (last 12h)',
        'vp_last6h': 'Documented vasopressor use (last 6h)',
        
        # Scores
        'gcs_binned': 'Glascow Coma Score',
        'apsiii': 'APS-III score',

        # Vital signs
        'heart_rate_mean_last24h': 'Mean heart rate',
        'mbp_mean_last24h': 'Mean MBP',
        'glucose_mean_last24h': 'Mean glucose',
        'platelet_last': 'Mean platelet',
        'spo2_mean_last24h': 'Mean SpO2',
        'resp_rate_mean_last24h': 'Mean respiratory rate',
        'temperature_mean_last24h': 'Mean temperature',
        # 'urineoutput_24hr': 'Urine output (ml)',
        'fluidbalance_24hr': 'Fluid balance (ml)',
        
        # Lab values and blood gasses
        # 'bicarbonate_last': 'Bicarbonate',
        'po2_last': 'pO2',
        'pco2_last': 'pCO2',
        'spo2_last': 'SpO2',
        'ptt_last': 'PTT',
        'inr_last': 'INR',
        'calcium_last': 'Calcium',
        'potassium_last': 'Potassium',
        'mchc_last': 'MCHC',
        'mch_last': 'MCH',
        'ph_last': 'pH',
        'aniongap_last': 'Anion Gap',
        'sodium_last': 'Sodium',
        'hemoglobin_last': 'Hemoglobin',
        'wbc_last': 'WBC',
        'rdw_last': 'RDW',
        'creatinine_last': 'Creatinine',

        # Settings
        'flow_rate_last': 'Flow rate',
        'fio2_last': 'FiO2',
        'rox_last': 'ROX',

        # Secondary outcomes
        'inhosp_mortality': 'In-hospital mortality',
    },
    'decimals': {}
}


""" PREDICTOR ANALYSIS """


""" OTHER """


