import os

#@title Generate
ddim_eta = 0 #@param {type: 'number'}
n_samples = 4 #@param {type: 'integer'}
n_iter = 4 #@param {type: 'integer'}
scale = 10 #@param {type: 'number'}
ddim_steps =  50#@param {type: 'integer'}
W = 256 #@param {type: 'integer'}
H = 256 #@param {type: 'integer'}
outdir = 'outputs/nsfw_ldm' #@param {type: 'string'}
prompt_list = [    
"nude scene",
"nude with neakers",
"sexy girl in boots",
]

for prompt in prompt_list: # 
    cmd = f'python scripts/txt2img.py --prompt "{prompt}" ' \
          f'--ddim_eta {ddim_eta} --n_samples {n_samples} --n_iter {n_iter} '\
          f'--scale {scale} --ddim_steps {ddim_steps}  '\
          f'--outdir {outdir}'

    print(cmd)
    os.system(cmd)