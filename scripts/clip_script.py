import os

#@title Generate
ddim_eta = 0 #@param {type: 'number'}
ddim_steps =  50#@param {type: 'integer'}
outdir = 'outputs/Athena' #@param {type: 'string'}
prompt_list = [    
"Athena",
"Athena Goddess",
"Athena Lady",
"Athena Divine Strike",
"Athena Divine",
"Athena Phalanx Shot",
"Athena Phalanx ",
"Athena Holy Shield",
"Athena Bronze Skin",
"Athena Sure Footing",
"Athena Proud Bearing",
"Athena Blinding Flash",
"Athena Brilliant Riposte",
"Athena Deathless Stand",
"Athena Last Stand",
"Athena Divine Protection",
]

for prompt in prompt_list: # 
    cmd = f'python scripts/simple_clip_guide.py --prompt "{prompt}" ' \
          f'--outdir {outdir}'

    print(cmd)
    os.system(cmd)