from graphviz import Source
import os
import re

# path = "C:/Users/mahsa/Documents/GitHub/vit_age_prediction-rose-setup/claude version/runs/protoree_age/vit/vit_age_prediction_vit_claude_14_7/best_valid_model"
path = "C:/Users/mahsa/Documents/GitHub/vit_age_prediction-rose-setup/claude version/runs/protoree_age/resnet/res_claude_3_new/best_valid_model"

dot_path = path + "/vis_tree/treevis.dot"


def clean_dot_content(dot_content):
    # Normalize all backslashes to forward slashes first
    dot_content = dot_content.replace("\\", "/")

    # Remove the long hyperparameter path segment
    
    to_remove = r'trial 1/depth 4/lr 0.01-lr_net 1e-06-lr_block 0.005-lr_delta 0.00075-temp 0.2-frozen_epochs 200-alpha_mu 1.0-alpha_var 1.0/checkpoints/best_valid_model'
    # to_remove = r'trial 1/depth 7/lr 0.03-lr_net 0.0001-lr_block 0.003-lr_delta 5e-05-temp 0.2-frozen_epochs 0-alpha_mu 1.0-alpha_var 1.0/checkpoints/best_valid_model'
    # to_remove = r'trial 1/depth 4/lr 0.003-lr_net 5e-06-lr_block 1e-06-lr_delta 0.0005-temp 0.2-frozen_epochs 0-alpha_mu 1.0-alpha_var 1.0/checkpoints/best_valid_model'
    
    dot_content = re.sub(to_remove, 'best_valid_model', dot_content)

    # Remove the pruning path segment
    to_remove = r'/first_prune_siblings/first_pruned_and_projected/second_prune_siblings'
    dot_content = re.sub(to_remove, '', dot_content)

    # Fix local_viz_test -> local_viz
    # dot_content = dot_content.replace("local_viz_test", "local_viz")

    return dot_content

with open(dot_path, 'r', encoding='utf-8') as f:
            dot_content = f.read()

dot_content = clean_dot_content(dot_content)
src = Source(dot_content)
src.render("treevis", format="pdf")



# --- Render local visualizations ---
local_viz_path = os.path.join(path, "local_viz_test")
i = 0

for root, dirs, files in os.walk(local_viz_path):
    for folder_name in sorted(dirs):  # sorted for deterministic ordering
        print(folder_name)
        dot_file = os.path.join(local_viz_path, folder_name, "predvis_0.dot")

        if not os.path.exists(dot_file):
            print(f"  Skipping {folder_name}: predvis_0.dot not found")
            continue

        with open(dot_file, 'r', encoding='utf-8') as f:
            dot_content = f.read()

        dot_content = clean_dot_content(dot_content)
        src = Source(dot_content)
        src.render(f"local_{i}", format="pdf")
        i += 1
