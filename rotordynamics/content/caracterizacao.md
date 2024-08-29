# Caracterização do sistema

Tendo como base os dados numéricos do segundo capítulo do livro texto da disciplina, "Rotordynamics Prediction in Engineering, Second Edition", {cite:ts}`lalanne_rotordynamics_1998`, foram replicadas as análises apresentadas no livro e expandidas.

Para a modelagem do sistema dinâmico foi criada uma bliblioteca em Python denominada `rotor_analysis` composta por três módulos, nomeados como `utilities.py`,`rotordynamics.py` e `results.py`, utilizando-se o paradigma de programação orientado a objeto (OOP - Object Oriented Paradigm). A biblioteca pint foi utilizada para manipular grandezas físicas de forma consistente bem como tornar a leitura do código mais autoexplicativa.

O  módulo `utilities.py` fornece classes para modelar materiais e objetos geométricos como cilindros e discos. Ele calcula as diversas propriedades físicas, como volume, massa, área de superfície e momentos de inércia.

Classes:

    Material: Representa um material com um nome e densidade.
    Cylinder: Representa um cilindro oco e calcula suas propriedades.
    Collection: Uma classe para manusear uma coleção de objetos (discos).
    
Já o  módulo `rotordynamics.py` fornece classes para modelar um rotor girando com uma massa desbalanceada.

Classes:

    Disc: Representa um disco, herdando de Cylinder.
    Shaft: Representa um eixo, herdando de Cylinder.
    Rotor: Representa um conjunto rotativo com um eixo e discos.