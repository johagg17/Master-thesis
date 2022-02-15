
"""
    In order to run this python file do following:
        
        Change the datapath on line 39 to the path where the folder is. 
        Navigate to this .py file in the terminal, then run 
        bokeh serve --show admission.py
        
        Now a browser will pop up and start load, at first it will take some time to load in everything, this 
        is due to reading and processing prescriptions.csv. 
        
        When the loading is done, you should be able to see some selections and tabs that you can navigate
        between. 
"""

"""
    Import bokeh libraries

"""
from bokeh.models import ColumnDataSource, Slider, NumeralTickFormatter, HoverTool, WheelZoomTool, ResetTool, PanTool
from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import widgetbox, column, row, grid
from bokeh.plotting import figure, show, output_file, curdoc
from bokeh.io import show
from bokeh.models import CustomJS, Select, TableColumn, DataTable
from bokeh.io import output_notebook
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.plotting import figure, curdoc
from bokeh.models import DatetimeTickFormatter

"""
    Other libraries
"""
import pandas as pd
import random
import numpy as np


parent_path = 'D:\skolaÅr5\MasterThesis\mimic-iv-1.0'


class VisData(object):
    
    def __init__(self, parent_path, appointments, n_patients):
        self.path = parent_path
        self.nappointments = appointments
        self.npatients = n_patients
        
        self.__extract()
        
    
    def __extract(self):
        
        parent_path = self.path
        
        # Read patient and admission tables
        df_patients = pd.read_csv(parent_path + '\core\patients.csv')
        df_admissions = pd.read_csv(parent_path + r'\core\admissions.csv')
        
        # Merge patients with admissions
        df_patadm = pd.merge(df_patients, df_admissions, on='subject_id')
        
        # Extract patients with admissions > n admissions
        admperpat = df_patadm.groupby(['subject_id']).size().reset_index(name='count')
        more_than_5appointments = admperpat[admperpat['count'] > self.nappointments]

        patientsids = more_than_5appointments.sample(n=self.npatients)['subject_id'].tolist()

        df_final = df_patadm[df_patadm['subject_id'].isin(patientsids)].sort_values('admittime')

        df_final['vis_label'] = 1

        df_final['admittime'] = pd.to_datetime(df_final['admittime'])
        df_final['dischtime'] = pd.to_datetime(df_final['dischtime'])
        age_firstadm = df_final['anchor_age'].iloc[0]
        year_firstadm = df_final['admittime'].iloc[0].year
        df_final['time_stay'] = abs(df_final['admittime'] - df_final['dischtime']).apply(lambda x: x.days)
        df_final['age'] = df_final['admittime'].apply(lambda x: abs(x.year - year_firstadm) + age_firstadm)
    
        self.df_final = df_final
        
        
        # Read code and diagnose table
        codes = pd.read_csv(parent_path + r'/hosp/d_icd_diagnoses.csv')
        pat_diagnoses = pd.read_csv(parent_path + r'/hosp/diagnoses_icd.csv')
        
        df_diagcodes = pat_diagnoses.merge(codes, on='icd_code', how='inner')
        
        self.diagnosedata = df_diagcodes
        
        
        # Read prescriptions table
        prescriptions = pd.read_csv(parent_path + '/hosp/prescriptions.csv', dtype={'NDC': 'category'})
        prescriptionsdropped=prescriptions.drop(columns=['drug_type', 'gsn', 'pharmacy_id', 'drug_type', 'form_rx', 'form_unit_disp', 'route', 'form_val_disp', 'doses_per_24_hrs'])
        
        self.prescriptions = prescriptionsdropped
        
        
    def getData(self):
        return self.df_final
    
    
    def getUniquepat(self):
        
        patids = self.df_final['subject_id'].unique().tolist()
        patids = list(map(str, patids))
        
        return patids
    
    def getAdmissions(self, patient_id):
        patientdata = self.df_final[self.df_final['subject_id'] == int(patient_id)]
        
        return patientdata
    
    def getMedications(self, patient_id, admission_id):
        
        pres = self.prescriptions[self.prescriptions['subject_id'] == int(patient_id)]
        pres = pres[pres['hadm_id'] == int(admission_id)]
        
        return pres
    
    def getDiagnoses(self, patient_id, admission_id):
        
        pat_data = self.diagnosedata[self.diagnosedata['subject_id'] == int(patient_id)]
        adm_data = pat_data[pat_data['hadm_id'] == int(admission_id)][['icd_code', 'long_title']]
        
        return adm_data
        
visObject = VisData(parent_path, 5, 100)


unique_patients = visObject.getUniquepat()

tooltips = [('Age', '@age'), 
            ('Admission Stay', '@time_stay'),
           ('Admission type', '@admission_type')]

#hovtool = HoverTool(tooltips=tooltips)

tools = [HoverTool(tooltips=tooltips), WheelZoomTool(), ResetTool(), PanTool()]
p = figure(title = "Admissions for patient {} ".format(unique_patients[0]), plot_width=800, plot_height=350, tools=tools)

p.xaxis.formatter=DatetimeTickFormatter(
    hours=["%d %B %Y"],
    days=["%d %B %Y"],
    months=["%d %B %Y"],
    years=["%d %B %Y"],
)

initdata = visObject.getAdmissions(int(unique_patients[0])) 

source = ColumnDataSource(data=initdata)

p.rect(x='admittime', y='vis_label', width=3, height=20, color="#CAB2D6",
      width_units="screen", height_units="screen", source=source)



unique_adm = list(map(str, initdata['hadm_id'].unique()))

pat_select = Select(title="Patients:", value=str(unique_patients[0]), options=unique_patients)
adm_select = Select(title="Admissions:", value=str(unique_adm[0]), options=unique_adm)



dff = visObject.getDiagnoses(unique_patients[0], unique_adm[0])  
df_med = visObject.getMedications(unique_patients[0], unique_adm[0])  

diagnosesource = ColumnDataSource(dff)
medsource = ColumnDataSource(data=df_med)


diagnosecolumns = [
    TableColumn(field='icd_code', title='ICD code'),
    TableColumn(field='long_title', title='Description')
]

medcolumns = [
    TableColumn(field='starttime', title='start time'),
    TableColumn(field='stoptime', title='end time'),
    TableColumn(field='ndc', title='ndc code'),
    TableColumn(field='drug', title='drug'),
    TableColumn(field='dose_val_rx', title='dose'),
    TableColumn(field='dose_unit_rx', title='unit')
]

diagnosetable = DataTable(source=diagnosesource, columns=diagnosecolumns, width=800)
prescriptionstable = DataTable(source=medsource, columns=medcolumns, width=800)

def adm_select_callback(attr, old, new):
    # Change value of current
    adm_select.value = str(new)
    
    # Change diagnosetable
    newdata = visObject.getDiagnoses(pat_select.value, new)  
    diagnosetable.source.data = newdata
    
    # Change medtable
    df_med = visObject.getMedications(pat_select.value, new)  
    prescriptionstable.source.data = df_med
    
    

def pat_select_callback(attr, old, new):
    
    # Change figure title
    p.title.text = "Admissions for patient {} ".format(new)
    # Change value of current
    pat_select.value = str(new)
    
    # Admission figure
    data = visObject.getAdmissions(new)
    unadm = list(map(str, data['hadm_id'].unique().tolist()))
    source.data = data
    adm_select.options = unadm
    adm_select.value = unadm[0]
    
    # Diagnosetable
    diagnosedata = visObject.getDiagnoses(new, adm_select.value)
    diagnosetable.source.data = diagnosedata
    
    # Meddata
    df_med = visObject.getMedications(new, adm_select.value)  
    prescriptionstable.source.data = df_med
    
    
    
pat_select.on_change('value', pat_select_callback)
adm_select.on_change('value', adm_select_callback)

tab1 = Panel(child=p, title="Admissions")
tab2 = Panel(child=diagnosetable, title="Diagnoses")
tab3 = Panel(child=prescriptionstable, title="Prescriptions")

tabs = Tabs(tabs=[tab1, tab2, tab3])

l = grid([row(pat_select, adm_select), tabs])



curdoc().add_root(l)


