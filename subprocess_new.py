# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:08:20 2024

@author: EMKA
"""

import subprocess

# Run a simple shell command
result = subprocess.run(['echo', 'Hello, World!'], capture_output=True, text=True)

# Print the output
print(result.stdout)