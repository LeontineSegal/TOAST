# %% load modules
from typing import Optional

# %%

def convert_name_mol_line(name_mol_line: str, from_file: Optional[str] = 'fits', to_file: Optional[str] = 'radex', dataset : Optional[str] = 'horsehead') -> str:

    if dataset == 'horsehead' :
        mol_line_fits = [
        '12co10',
        '13co10', '13CO(2-1)',
        'c18o10', 'C18O(2-1)',
        'hcop10',
        'h13cop10', 
        ]

        mol_line_radex = [
        '12co(1-0)',
        '13co(1-0)', '13co(2-1)',
        'c18o(1-0)', 'c18o(2-1)',
        'hcop(1-0)',
        'h13cop(1-0)', 
        ]
        
        mol_line_latex = [
        r'$^{12}\mathrm{CO}(1{-}0)$',
        r'$^{13}\mathrm{CO}(1{-}0)$', r'$^{13}\mathrm{CO}(2{-}1)$',
        r'$\mathrm{C}^{18}\mathrm{O}(1{-}0)$', r'$\mathrm{C}^{18}\mathrm{O}(2{-}1)$',
        r'$\mathrm{HCO}^{+}(1{-}0)$',
        r'$\mathrm{H}^{13}\mathrm{CO}^{+}(1{-}0)$', 
        ]
        
    else : 
        # to do, following your dataset 
        pass

    format = ['fits', 'radex', 'latex']
    names = [mol_line_fits, mol_line_radex, mol_line_latex]

    idx_from_file = format.index(from_file)
    list_from_file = names[idx_from_file]
    idx_to_file = format.index(to_file)
    list_to_file = names[idx_to_file]

    return list_to_file[list_from_file.index(name_mol_line)]


