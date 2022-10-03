# EMG_EEG_EXO
## Huom!!

Matplotlib 3.3.2 and above versions do not print the figures, since set_xtrick and set_xticklabels are not supported. In order to print images you need to downgrade matplotlib version to 3.2.0

### Function to downgrade matplotlib
from ipython console = pip install matplotlib==3.2.0 --user 

<div class="center">

```mermaid
graph TD
    A[from ipython console] --> B(pip install matplotlib==3.2.0 --user)
    B --> C[Restart]
    C --> D[pip install matplotlib==3.2.0]
```

from command prompt  = conda install matplotlib==3.2.0
<br />

```mermaid
graph TD
    A[from command prompt] --> B(conda install matplotlib==3.2.0)
    
```

