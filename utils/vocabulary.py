import numpy as np

def write_age_to_file(age_list, file_path) -> None:
    """ Function to write age vocabulary file """
    """ age_list will be 2d, where outer index will be patient, and inner index is the patient's visits"""
    
    all_ages = []
    
    ## [[a1, a2], [a1, a2], [a1, a2, a3, a4], [a1, a2, a3]]
    
    for patient_ages in age_list:
        all_ages.extend(patient_ages)
        
    all_agesnp = np.array(all_ages)
    np.save(file_path, all_agesnp)
    
    
def write_codes_to_file(code_list, file_path) -> None: 
    """ Function used for writing processing list_ which can contain either Snomed codes, ccssr codes, icd codes """
    """ This function will be integrated with additional features later.. such as writing multiple code formats into one file (diagnosis codes, medication codes)"""
    
    all_codes = []
    
    ## [[[d1, d2], [d3, d5]], [[d1, d2]]]
    
    for patient in code_list:
        for patient_visit in patient:
            all_codes.extend(patient_visit)
            
    all_codes = list(set(all_codes))
    all_codes = np.array(all_codes)
    np.save(file_path, all_codes)
    