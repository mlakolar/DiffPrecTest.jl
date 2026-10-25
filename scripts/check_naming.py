#!/usr/bin/env python3
"""Check new Julia declarations; compatibility files and mathematical names are explicit exceptions."""
import argparse
from pathlib import Path
import re

# Matrix caches and scientific notation retain their conventional identifiers.
MATHEMATICAL = {'A','B','C','D','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
                'Ax','At','Bt','Xt','Yt','XtY','XtX','sUtY','Ws','SX','SY','Xn','Yn','SXn','SYn',
                'Xa','Ya','SXa','SYa','Xr','Yr','posA','posR','T0','F','Sx','Sy','Sxy','Sxx','Syy','AD','Vt','AU','D_calls','Dt_calls','Dx','Dxp','Dty','Dtyp','Bpath'}
FUNCTION = re.compile(r'^\s*(?:function\s+)([\w.!]+)\s*\(|^\s*([\w.!]+)\([^\n]*\)\s*(?:where\s+[^=]+)?=(?!=)')
FIELD = re.compile(r'^\s*(\w+)\s*::')
LEGACY_KEYWORDS = {'printEvery','normD','maxK','requested_maxK','epsTol','maxIter','checkEvery','bootSamples','estimSupport'}
FACTORIES = {'IndBallLinf','Constrained','_AtomIterate','CDLeastSquaresLoss','CDWeightedLSLoss','CDSqrtLassoLoss','CDSqrtQuadraticLoss','CDQuadraticLoss'}
TYPE = re.compile(r'^\s*(?:mutable\s+)?(?:struct|abstract\s+type|primitive\s+type)\s+(\w+)')

def check(root):
    errors=[]
    types={m.group(1) for path in (root/'src').rglob('*.jl')
           for line in path.read_text().splitlines() if (m := TYPE.match(line))}
    types |= {m.group(1) for path in (root/'src').rglob('*.jl') for m in re.finditer(r'^\s*const\s+(\w+)\s*=',path.read_text(),re.M)}
    for path in sorted((root/'src').rglob('*.jl')):
        if 'deprecated' in path.stem: continue
        in_doc=False; in_struct=False
        lines=path.read_text().splitlines()
        for number,line in enumerate(lines,1):
            if line.count('"""') % 2: in_doc=not in_doc; continue
            if in_doc: continue
            line=line.split('#',1)[0]
            typ=TYPE.match(line)
            if typ:
                name=typ.group(1)
                if not name.lstrip('_')[0].isupper():errors.append((path,number,'type',name))
                in_struct='struct' in line
                continue
            if re.match(r'^\s*end\b',line):in_struct=False
            function=FUNCTION.match(line)
            if function:
                name=function.group(1) or function.group(2)
                # Qualified names implement external interfaces; constructors must name an actual type.
                if '.' not in name and name not in types and name not in FACTORIES and name not in MATHEMATICAL and re.search('[A-Z]',name):
                    errors.append((path,number,'function',name))
                if ';' in line or line.lstrip().startswith('function '):
                    signature=line[line.find('(')+1:]
                    depth=1 + signature.count('(')-signature.count(')')
                    offset=number
                    while depth>0 and offset<len(lines):
                        more=lines[offset].split('#',1)[0]
                        signature+=' '+more
                        depth+=more.count('(')-more.count(')')
                        offset+=1
                    # Only the top level keyword list; ignore nested constructor defaults.
                    depth=0; keyword=False; segment=''
                    for char in signature:
                        if char in '([{': depth+=1
                        elif char in ')]}':
                            if depth==0: char=','
                            else: depth-=1
                        if char==';' and depth==0:
                            keyword=True; segment=''; continue
                        if char==',' and depth==0:
                            if keyword:
                                match=re.match(r'\s*(\w+)\s*(?:::[^=]+)?=',segment)
                                if match:
                                    key=match.group(1)
                                    if re.search('[a-z][A-Z]',key) and key not in LEGACY_KEYWORDS:
                                        errors.append((path,number,'keyword',key))
                            segment=''
                        else: segment+=char
            field=FIELD.match(line) if in_struct else None
            if field:
                name=field.group(1)
                if re.search('[A-Z]',name) and name not in MATHEMATICAL:
                    errors.append((path,number,'field',name))
    return errors

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packages',nargs='*',type=Path)
    args=parser.parse_args()
    roots=args.packages or [Path(__file__).resolve().parents[1]]
    errors=[item for root in roots for item in check(root)]
    for path,line,kind,name in errors: print(f'{path}:{line}: noncanonical {kind}: {name}')
    if errors: raise SystemExit(1)
    print(f'Naming declarations checked in {len(roots)} package(s).')
