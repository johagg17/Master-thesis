
import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf,col

class PatientProcessor(object):
    
    def __init__(self, parent_path, **files):
        
        self.parent_path = parent_path
        self.files = files
        self.pat_data = None
        self.__process__()

    def __init_spark__(self):
        
        spark = SparkSession \
            .builder \
            .appName("Python Spark SQL basic example") \
            .config("spark.some.config.option", "some-value") \
            .getOrCreate()
        return spark


    def __process__(self):
        spark = self.__init_spark__()
        parent_path = self.parent_path
        
        codes_icd_path = parent_path + '/hosp/diagnoses_icd.csv'
        diagnose_codes_icd = spark.read.load(codes_icd_path, format="csv", sep=",", inferSchema="true", header="true")
        
        diagnoses_codes_path = parent_path + '/hosp/d_icd_diagnoses.csv'
        diagnoses_codes = spark.read.load(diagnoses_codes_path,  format="csv", sep=",", inferSchema="true", header="true")  
        
        patients_path = parent_path + '/core/patients.csv'
        patients = spark.read.load(patients_path,  format="csv", sep=",", inferSchema="true", header="true")  
        
        admission_path = parent_path + '/core/admissions.csv'
        admissions = spark.read.load(admission_path,  format="csv", sep=",", inferSchema="true", header="true")  
        
        admissions_ = admissions.withColumnRenamed('subject_id', 'subject_id2')
        patients_admissions = patients.join(admissions_, patients.subject_id == admissions_.subject_id2, 'inner').drop('subject_id2')
        
        diagnoses_codes_ = diagnoses_codes.withColumnRenamed('icd_code', 'code').withColumnRenamed('icd_version', 'version')

        patients_diagnoses = diagnose_codes_icd.join(diagnoses_codes_, (diagnose_codes_icd.icd_code == diagnoses_codes_.code) \
                                                     & (diagnose_codes_icd.icd_version == diagnoses_codes_.version), "inner") \
        .drop('code', 'version')
        
        patients_diagnoses = patients_diagnoses.withColumnRenamed('subject_id', 'sub2').withColumnRenamed('hadm_id', 'h_id')
        
        
        pat_data = patients_admissions.join(patients_diagnoses, ((patients_admissions.subject_id == patients_diagnoses.sub2) & \
                                                                   (patients_admissions.hadm_id == patients_diagnoses.h_id)) ,\
                                              'inner').drop('sub2', 'h_id')
        
        needed_columns = set(['subject_id', 'anchor_age', 'hadm_id', 'admittime', 'dischtime', 'icd_code', 'icd_version', 'long_title'])
        
        columns_ = [c for c in pat_data.columns if c not in needed_columns]
        
        new = pat_data.drop(*columns_).na.drop()
        
        self.pat_data = new
    
    def getData(self):
        return self.pat_data
        
    def write_data(self, path):
        
        if path_data is None:
            return None
        
        self.path_data.write.option('header', True).csv(path)
        
        
        
        
        
        
        
        
        

if __name__ == "__main__":
    parent_path = 'D:\skolaÅr5\MasterThesis\mimic-iv-1.0'.replace('\\', '/') 


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
    extr = PatientProcessor(parent_path, **config)



