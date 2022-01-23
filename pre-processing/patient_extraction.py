
import pandas as pd

class PatientExtractor(object):
    
    def __init__(self, **files):
        
        self.files = files
        self.__subject_id_age__()

    def __subject_id_age__(self):
        assert self.files['id_age'],  "Subject_id and age file not included" 

        df_pat = pd.read_csv(self.files['id_age'])

        df_adm = pd.read_csv(self.files['pat_adm'])
        #subject_age = df_pat[df_pat['subject_id'], df_pat['anchor_age']]
        #print(subject_age)
        
        tot = pd.merge(df_pat, df_adm, on='subject_id')

        print(tot)


        



if __name__ == "__main__":
    parent_path = 'D:\skolaÅr5\MasterThesis\mimic-iv-1.0'.replace('\\', '/') 

    #print("Parent path: {}".format(parent_path))

    patients_f = parent_path + '/core/patients.csv'
    admission_f = parent_path + '/core/admissions.csv'
    pat_diagnos = parent_path + '/hosp/diagnoses_icd.csv'
    meaning_diagnos = parent_path + '/hosp/d_icd_diagnoses.csv'


    config = {
        'id_age': patients_f, 
        'pat_adm':admission_f,
        'pat_diagnose_icd': pat_diagnos, 
        'diagnos_meaning': meaning_diagnos,
        }
    extr = PatientExtractor(**config)




