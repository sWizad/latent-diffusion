import os

#@title Generate
seed = 1
outdir = 'outputs/greek_goddess' #@param {type: 'string'}
prompt_list = [    
"Goddess Persephone",
]

for prompt in prompt_list: # 
    cmd = f'python scripts/splite_pyramid.py --prompt "{prompt}" ' \
          f'--outdir {os.path.join(outdir,prompt.replace(" ", "-"))}_{seed} ' \
          f'--seed {seed} '

    print(cmd)
    os.system(cmd)