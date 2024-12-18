import pickle
AllData = {}
with open('input/default.p', 'rb') as handle:
    AllData = pickle.load(handle)

Sum = 0
SumHadron = 0
SumJet = 0
for i in AllData['data']['HeavyIon']['R_AA']:
    Sum += AllData['data']['HeavyIon']['R_AA'][i]['x'].shape[0]
    if 'Jet' in i:
        SumJet += AllData['data']['HeavyIon']['R_AA'][i]['x'].shape[0]
    else:
        SumHadron += AllData['data']['HeavyIon']['R_AA'][i]['x'].shape[0]

print(Sum, SumJet, SumHadron)
