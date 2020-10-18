labels = []
for i in range(len(json_files)):
    with open(files[i], 'r+') as f:
        data=f.read()
        # parse file
        p = json.loads(data)
        p = p[['publication_date','decision','notice_no','CPV','project_id','project_title','project_details','type']].copy()
        json.dump(p, f)

with open(files[0], 'r+') as f:
    data=f.read()
    # parse file
    v = json.loads(data)

    del p['label']
    p = json.loads(data)  

with open(files[1], 'r') as handle:
    json_data = [json.loads(line) for line in handle]




import json

file = 'ID_97050_NO_964393.json'

path1 = './test_json/' + file
path2 = './test_json_new/' + file

with open(path2, 'r') as f:
    data=f.read()
    p = json.loads(data)
    p['label'] = 'yes'
    with open(path2, 'w') as outfile:
        json.dump(p, outfile) # save labeled project 