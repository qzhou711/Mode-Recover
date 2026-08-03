import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();rows=[]
for lane in sorted(a.root.glob('width_*')):
    for f in sorted(lane.glob('epoch*/eval120/metrics.json')):
        d=json.loads(f.read_text());r={'lane':lane.name,'epoch':int(f.parts[-3].replace('epoch','')),'success_rate':d['success_rate'],'coverage':d['unique_successful_modes'],'entropy':d['normalized_mode_entropy'],'metrics':str(f)};r['passed_gate']=r['success_rate']>=.80 and r['coverage']>=22;rows.append(r)
eligible=[r for r in rows if r['passed_gate']];pool=eligible or rows;selected=max(pool,key=lambda r:(r['success_rate'],r['coverage'],r['entropy'],-r['epoch']))
report={'protocol':'Standard-120 gate SR>=80%, coverage>=22; maximize SR, coverage, entropy','rows':rows,'any_passed':bool(eligible),'selected':selected};(a.root/'selection.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
