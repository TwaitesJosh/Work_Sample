import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize


def match_to_number(match):
    if match == 'exactMatch':
        return 0
    elif match == 'closeMatch':
        return 1
    elif match == 'narrowMatch':
        return 2
    elif match == 'broadMatch':
        return 3
    elif match == 'relatedMatch':
        return 4
    elif match == 'wrongMatch':
        return 99
    else:
        return 99
    
def keep_best_match(tab):
    tab['Match Number'] = tab.apply(lambda x:match_to_number(x['Type of Match']), 1)
    if len(tab)>0:
        return tab.sort_values('Match Number').iloc[0]
    else:
        return tab.iloc[0]


class dataloader:

    def __init__(self):
        self.occs = pd.read_csv('/Users/joshuatwaites/SUSSEX_MCC/backend/Occs_processed.csv')
        self.embeddings = np.load('/Users/joshuatwaites/SUSSEX_MCC/BERT.npy')
        self.skills = pd.read_csv('/Users/joshuatwaites/SUSSEX_MCC/skills_en.csv')

def load_data():
    data = dataloader()
    return data.occs, data.embeddings, data.skills

def create():
    return 0

def load_files():
    skills = pd.read_csv('skills_en.csv')
    occs = pd.read_csv('occupations_en.csv')
    occupation_skills = pd.read_csv('occupationSkillRelations.csv')
    embeddings = np.load('BERT.npy')
    onet_skills = pd.read_csv('/Users/joshuatwaites/Downloads/ONET (Occupations)_0.csv', skiprows = 16, header = 1)

    
def create_green():
    green = pd.read_csv('/Users/joshuatwaites/Downloads/ESCO_dataset_csv/greenSkillsCollection_en.csv')
    green_score_essential = [len(pd.merge(skills.iloc[occs['Essential Skills'].iloc[i]], green)) for i in np.arange(len(occs))]
    green_score_optional = [len(pd.merge(skills.iloc[occs['Optional Skills'].iloc[i]], green)) for i in np.arange(len(occs))]
    occs['Green_Score_Essential'] = green_score_essential
    occs['Green_Score_Optional'] = green_score_optional

def create_digital():
    digital = pd.read_csv('/Users/joshuatwaites/Downloads/ESCO_dataset_csv/digitalSkillsCollection_en.csv')
    digital_score_essential = [len(pd.merge(skills.iloc[occs['Essential Skills'].iloc[i]], digital)) for i in np.arange(len(occs))]
    digital_score_optional = [len(pd.merge(skills.iloc[occs['Optional Skills'].iloc[i]], digital)) for i in np.arange(len(occs))]
    occs['Digital_Score_Essential'] = digital_score_essential
    occs['Digital_Score_Optional'] = digital_score_optional

def create_work_area():
    occs['Work Area'] = occs['Work Context']


    
def create_skills():
    skills = pd.read_csv('skills_en.csv')
    occs = pd.read_csv('occupations_en.csv')
    occupation_skills = pd.read_csv('occupationSkillRelations.csv')
    
    skills['Index'] = skills.index.values
    occ_skill = pd.merge(occupation_skills, skills, left_on = 'skillUri', right_on = 'conceptUri')
    occs_skill_rel = pd.merge(occs, occ_skill.groupby('occupationUri')['Index'].apply(lambda x: list(x)), left_on = 'conceptUri', right_on = 'occupationUri')
    occs_skill_rel.rename(columns={'Index': 'Optional Skills'}, inplace=True)

    occ_skill = pd.merge(occupation_skills, skills, left_on = 'skillUri', right_on = 'conceptUri')
    occ_skill_ess = occ_skill[occ_skill['relationType'] == 'essential']

    occs_skill_rel = pd.merge(occs_skill_rel, occ_skill_ess.groupby('occupationUri')['Index'].apply(lambda x: list(x)), left_on = 'conceptUri', right_on = 'occupationUri')
    occs_skill_rel.rename(columns={'Index': 'Essential Skills'}, inplace=True)
    occs = occs_skill_rel

def create_work_context():
    occs = pd.read_csv('occupations_en.csv')
    onet_skills = pd.read_csv('/Users/joshuatwaites/Downloads/ONET (Occupations)_0.csv', skiprows = 16, header = 1)
    q = pd.merge(occs, onet_skills, left_on = 'conceptUri', right_on = 'ESCO or ISCO URI', how = 'left')
    unique_q = q.groupby('conceptUri').apply(keep_best_match)
    onet_work_context = pd.read_excel('/Users/joshuatwaites/Downloads/Work Context.xlsx', sheet_name='Work Context')
    unique_features = onet_work_context['Element Name'].unique()
    unique_features_id = onet_work_context['Element ID'].unique()
    x = np.argsort(unique_features_id)
    unique_features = unique_features[x]
    unique_features_id = unique_features_id[x]
    unique_onets = onet_work_context['O*NET-SOC Code'].unique()
    # Take the rows which correspond to the aggregated scores for each occupations' work context feature
    df_work_context = onet_work_context[onet_work_context['Scale ID'].isin(['CT','CX'])]
    # Dictionary mapping work context feature ID to integers (i.e., to the element in the work context vector)
    map_work_context_to_int = dict(zip(unique_features_id, range(len(unique_features_id))))
    # Empty arrays to hold the vectors
    work_context_vectors = np.zeros((len(unique_onets),len(unique_features)))
    # For each ONET occupation...
    for j in range(len(unique_onets)):
        # ...select all features
        dff = df_work_context[df_work_context['O*NET-SOC Code']==unique_onets[j]]
        for i, row in dff.iterrows():
            # Assign each work context vector element its value
            v = map_work_context_to_int[row['Element ID']]
            work_context_vectors[j,v] = row['Data Value']
    work_context_vectors_rescaled = work_context_vectors.copy()
    # All features are rated starting from 1
    min_val = 1 
    max_vals = []
    for j, element_id in enumerate(list(map_work_context_to_int.keys())):
        
        # Check which is the maximal value for the particular feature
        if 'CT' in onet_work_context[(onet_work_context['Element ID']==element_id)]['Scale ID'].to_list():
            max_val = 3
        else:
            max_val = 5
        
        max_vals.append(max_val)
        
        # Rescale 
        work_context_vectors_rescaled[:,j] = (work_context_vectors[:,j]-min_val)/(max_val-min_val)
    work_context_vectors_norm = np.zeros((len(unique_onets),len(unique_features)))
    for j in range(work_context_vectors_rescaled.shape[0]):
        work_context_vectors_norm[j,:] = normalize(work_context_vectors_rescaled[j,:].copy().reshape(1,-1))    
        
    unique_onets_df = pd.DataFrame(data={'onet_code': unique_onets, 'j': list(range(len(unique_onets)))})
    occs = pd.merge(unique_q, unique_onets_df, left_on = 'O*NET Id', right_on = 'onet_code', how = 'left')
    occs['Work Context'] = occs.apply(lambda x: np.repeat(np.NaN,57)  if np.isnan(x['j']) else work_context_vectors_norm[int(x['j'])] ,1)
   
    work_context = pd.DataFrame(np.vstack(occs['Work Context'].values))
    work_context['conceptUri'] = occs['conceptUri']
    work_context.to_csv('work_context.csv', index = False)



def create_bert_skills():
    skills = pd.read_csv('/Users/joshuatwaites/SUSSEX_MCC/skills_en.csv')
    bert = np.load('/Users/joshuatwaites/SUSSEX_MCC/BERT.npy')
    bert_tab = pd.DataFrame(bert)
    bert_tab['skillUri'] = skills['conceptUri']
    bert_tab.to_csv('BERT_skills.csv', index = False)

