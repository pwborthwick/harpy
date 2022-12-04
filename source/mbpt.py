from __future__ import division
import numpy as np

diagrams_hh = []

def nodalPairCount_hh(order):
    #return the number of pairs of nodes in Hugenholtz diagrams of 'order'
  
    return int(0.5 * order * (order- 1))

def nodalLineCount_hh(order):
    #return the total number of line connecting nodes in the diagram
    
    return 2 * order

def nodalPairs_hh(order):
    #return the nodal pairs in  diagram of 'order'
    
    pairs = []
    
    for p in range(order):
        for q in range(p+1, order):
            pairs.append([p, q])
            
    return pairs 

def validDiagrams_hh(diagrams, order):
    #perform checks on diagrams for validity
    
    verified = []
    passed = False
    
    #get list of nodal pairs
    nodalPair = nodalPairs_hh(order)
    pairCount = nodalPairCount_hh(order)

    #number of lines
    for d in diagrams:
    
        if sum(d) == 2 * order: 
        
            #check line count for diagram
            vertex = [0] * order
          
            #accummulate lines to each node
            for i in range(pairCount):
                vertex[nodalPair[i][0]] += d[i]
                vertex[nodalPair[i][1]] += d[i]
          
            #check correct number of lines at node
            if vertex == [4] * order:
          
                #number of lines at nodes verified check connected
                route = [0,1]
                node = 1
                tested = [False] * pairCount
                tested[0] = True
          
                #loop over nodes to be found
                while True:
          
                    #loop over nodal pairs
                    for i in range(pairCount):
            
                        pair = nodalPair[i]
              
                        #has pair been checked
                        if not tested[i]:
              
                            #is current node in nodal pair
                            if node == pair[0]: 
                                route.append(pair[1])
                                node = pair[1]
                            elif node == pair[1]: 
                                route.append(pair[0])
                                node = pair[0]
                            tested[i] = True
                                  
                    #all nodes tested leave while loop
                    if all(t == True for t in tested): break
                
                #have we got all nodes, make unique and sort
                route = list(set(route))
                route.sort()
                passed = True
                for i in range(order):
                    if i != route[i]: passed = False

                if passed: verified.append(d)
        
    return verified

def nodalPairConnectionsCombinations_hh(connections, nodePair, pairCount, order):
    #generate all combinations of connections between pairs
    
    #recursive return
    
    if nodePair == pairCount: 
        return validDiagrams_hh(connections,order)
    
    #define maximum number of connections
    limit = 3
    
    #adjust for special case order 2
    if order == 2: limit += 1
    
    #make copy of connections as we are modifying it and don't want to processed appended elements
    c = connections.copy()

    #loop over all elements in original connection list
    for connection in connections:    
    
        #loop over all possible connection types, 
        for i in range(1, limit+1):

            #make copy of current connection
            t = connection.copy()
            t[nodePair] = i

            #if sum of elements is less than or equal to allowed lines in diagram save
            if sum(t) <= 2 * order:
                c.append(t)
                
    #don't need original list now
    del connections
        
    #increment the nodePair
    nodePair += 1
      
    #recurse
    node = nodalPairConnectionsCombinations_hh(c, nodePair, pairCount, order)
        
    return node

def verifyArrow_hh(up, pairs, diagram, order):
    #check up arrow combination compatible with original diagram

    nodes = np.zeros(order)

    #loop over all pairs of nodes
    for n, pair in enumerate(pairs):
        i = pair[0]
        j = pair[1]

        #sum up arrows at each node
        if i < j:
            nodes[j] += up[n]
            nodes[i] += diagram[n] - up[n]
        else:
            nodes[i] += up[n]
            nodes[j] += diagram[n] - up[n]

    passed = True
    for i in range(order): 
        if nodes[i] != 2: passed = False 

    return passed


def upArrow_hh(up, nodePair, diagram, pairs, pairCount, order):
    #up arrow combinations for diagram

    if nodePair == pairCount-1: 

        passed = verifyArrow_hh(up, pairs, diagram, order)
        if passed: diagrams_hh.append([diagram,up.copy()])

        return 

    nodePair += 1

    #get limits of up connections
    lo = max(diagram[nodePair] - 2, 0)
    hi = min(diagram[nodePair],2)

    #generate combination within range
    for i in range(lo, hi+1):

        up[nodePair] = i
        upArrow_hh(up, nodePair, diagram, pairs, pairCount, order)

    nodePair -= 1

    return 

def upArrowCombinations_hh(diagramCombinations,order):
    #get all combinations of up arrows

    for i in diagramCombinations:

        nodePair = -1
        pairCount = nodalPairCount_hh(order)

        arrows = [0] * pairCount
        upArrow_hh(arrows, nodePair, i, nodalPairs_hh(order), pairCount, order) 


    return len(diagrams_hh)

def downArrow_hh(diagram, arrow, pairs):
    #compute the down arrows from an up arrow specification

    down = [0] * pairs
    for pair in range(pairs):
        down[pair] = abs(diagram[pair] - arrow[pair])
    

    return down

def connectionFlow_hh(up, down, order):
    #determine characteristics of each node

    #special case order 2
    if order == 2: return [[0,1,'d','i','a'],[0,1,'d','i','b'],[0,1,'u','o','r'], [0,1,'u','o','s'], \
                           [0,1,'d','o','a'],[0,1,'d','o','b'],[0,1,'u','i','r'], [0,1,'u','i','s']]

    def nodalFlow(pair, p, a, d, id, flows):
        #assign flow list element

        i, j= pair
        la = [i, j, a, d, id]
        id += 1

        #allow for two lines in same direction between same pair
        if p == 2: 
            lb = [i, j, a, d, id]
            id += 1
            flows.append(la)
            flows.append(lb)
        else: flows.append(la)

        return id, flows

    pairs = nodalPairs_hh(order)
    id = 1
    flowPattern = []

    for node in range(order):
        for i, pair in enumerate(pairs):

            #determine defining attributes of each line
            if node == pair[0] and up[i] != 0:
                id, flowPattern = nodalFlow(pair, up[i], 'u', 'o', id, flowPattern)
            if node == pair[0] and down[i] != 0:
                id, flowPattern = nodalFlow(pair, down[i], 'd', 'i', id, flowPattern)
            if node == pair[1] and up[i] != 0:
                id, flowPattern = nodalFlow(pair, up[i], 'u', 'i', id, flowPattern)
            if node == pair[1] and down[i] != 0:
                id, flowPattern = nodalFlow(pair, down[i], 'd', 'o', id, flowPattern)

    #rationalize numbering so each line has unique number
    connection = 0
    processed = []
    while connection != 2 * nodalLineCount_hh(order) :

        #current connection
        i,j,a,d,id = flowPattern[connection]

        #see if match in other direction
        for f in range(connection+1, len(flowPattern)):
            if flowPattern[f][:3] == [i,j,a] and flowPattern[f][3] != d :

                #if not already changed re-label and mark as changed
                if not f in processed:
                    flowPattern[f][4] = id
                    processed += [f]
                    break

        connection += 1

    #change to symbols
    holes =     ['a','b','c','d','e','f','g','h']
    particles = ['r','s','t','u','v','w','x','y']

    for i, line in enumerate(flowPattern):

        #get id and if digit translate to letter
        ID = line[4]
        if str(ID).isdigit():

            #determine if hole or particle
            if line[2] == 'd': id = holes[0]
            else: id = particles[0]

            #replace all occurences
            for j in range(i, len(flowPattern)):
                if flowPattern[j][4] == ID: flowPattern[j][4] = id

            #remove letter from pool
            if id in holes: del holes[0]
            else: del particles[0]
    
    return flowPattern

def rules_hh(flows, order):
    #evaluate rules

    rules = {}

    #rule 1 - in and out lines at each node
    eri = ''

    for i in range(order):

        #get this node values
        nodalFlow = flows[4*i: 4*i+4]

        #sort to get 'in' lines first        
        nodalFlow.sort(reverse = False, key=lambda i :i[3])

        eri += nodalFlow[0][4] + nodalFlow[1][4] + nodalFlow[2][4] + nodalFlow[3][4] + ','

    rules['doubleBars'] = eri[:-1]

    #rule 2 - mid-points of nodes

    levels = []
    
    for level in range(order-1):
        
        e = ''
        for f in flows:
            i,j,a,d,id = f

            #stop double counting by just taking 'in'
            if d == 'o': continue
            if i <=  level and j > level:

                if a == 'd': e += '+' + id
                else: e += '-' + id

        levels.append(e[1:])
    
    rules['orbitalEnergies'] = levels

    #rule 3 - down lines (h) and closed loops(l)

    h = 0
    for f in flows:
        if f[2] == 'd': h += 1

    #get string of labels
    eri = rules['doubleBars'].replace(',','')

    labels = []
    cycles = []

    current = 0

    while True:

        #target of cycle
        target = eri[current]

        labels.append(target)
        cycle = target + '->'

        #search string cyclically
        while True:

            current += 2
            next = eri[current]
            labels.append(next)

            cycle += next + '->'
            if target == next: break

            if next in eri[current+1:]:
                current = eri.index(next, current+1) 
            else:
                current = 0
                current = eri.index(next, current+1)

        #cycle finished
        cycles.append(cycle[:-2])
        cycle = ''

        #see if we've done all labels
        visited = True
        for i, a in enumerate(eri):
            if a not in labels: 
                visited = False
                break

        if visited: break

        #start where label still not visited
        current = i

    rules['sign'] = [h//2, len(cycles)]
    rules['cycles'] = cycles

    #rule 5 - equivalent lines

    equivalent = []
    for f in flows:
        if f[3] == 'i': equivalent.append(f)
    equivalent.sort()

    count = 0
    for i in range(len(equivalent)-1):
        if equivalent[i][:3] == equivalent[i+1][:3]:
            count += 1
  
    rules['powerTwo'] = -count

    return rules

def codeDiagrams_hh(molAtom, molBasis, molData, order):
    #produce python statements for diagrams of order

    diagrams_hh.clear()

    def getDenominators(rule, h):
        #generate reshapes for denominators
        
        code = '\nf = []'
        #get denominator strings
        f = []
        for r in rule.split(','):
            a = ''
            for s in r: 
                if not s in '+-': a += s
            f.append(a)

        #for each denominator expression generate reshape code
        for i, denominator in enumerate(f):

            d = '' 
            for j, a in enumerate(denominator):
                l = len(denominator)
   
                #occupied or virtual
                if a in h: d += '+ eocc'
                else: d += '- evir'

                if j != l-1:
                    d += '.reshape(-1' + ',1' * (l-j+-1) +') '

            d = '\nf.append( 1/(' + d[2:] + '))'

            code += d

        return code

    code = """
import rhf

eSCF = rhf.scf(molAtom, molBasis, molData, [])

from integral import buildFockMOspin
import numpy as np
ns = 2 * rhf.C.shape[0]
fs = buildFockMOspin(ns, rhf.C, rhf.fock)
efs = np.diag(fs)

from basis import electronCount
charge = molData['charge']
nsbf = len(molBasis)*2
nocc = int(electronCount(molAtom, charge))
nvir = nsbf - nocc

from integral import buildEriMO, buildEriDoubleBar
eriMO = buildEriMO(rhf.C, rhf.ERI)
MO = buildEriDoubleBar(nsbf, eriMO)

eocc = efs[:nocc]
evir = efs[nocc:]

o = slice(0,nocc)
v = slice(nocc, nsbf)

mp = 0.0
    """
    holes =     ['a','b','c','d','e','f','g','h']
    particles = ['r','s','t','u','v','w','x','y']

    #get combinations of diagrams
    connections = [[0] * nodalPairCount_hh(order)]
    nodePair = 0
    pairCount = nodalPairCount_hh(order)

    diagramCombinations = nodalPairConnectionsCombinations_hh(connections, nodePair, pairCount, order)

    #up arrow diagrams
    nDiagrams = upArrowCombinations_hh(diagramCombinations, order)

    for i, diagrams in enumerate(diagrams_hh):

        diagram, up = diagrams

        #down arrows
        down = downArrow_hh(diagram, up, pairCount)

        #arrow flow
        flows = connectionFlow_hh(up, down, order)

        #get rules
        rules = []
        rules.append(rules_hh(flows, order))

        #enumerate rules for diagrams

        for rule in rules:

            d = ''
            for a in rule['orbitalEnergies']:
                d += a.replace('+','') 
                d = d.replace('-','') + ','             

            auto = getDenominators(d[:-1], holes)            
            
            #sign and power of 1/2
            auto += '\nmp += '
            h = pow(-1, int(rule['sign'][0]) + int(rule['sign'][1]))
            if h<0: auto += '-1 * '
            auto += 'pow(1/2, ' + str(abs(rule['powerTwo'])) + ') * np.einsum(\''

            #double bar integrals and occupations
            auto += rule['doubleBars'] 

            d = ',' + d
            auto += d[:-1] + '\', MO['
            for a in rule['doubleBars']:
                if a in holes: auto += 'o,'
                elif a in particles: auto += 'v,'
                else :auto = auto[:-1] + '],MO['
            auto = auto[:-1] + '], '

            for i in range(order-1):
                auto += 'f[' + str(i) + '],'
            auto = auto[:-1] + ')'

        code += auto + '\n'
            
    data = {'molAtom':molAtom,'molBasis':molBasis,'molData':molData}

    exec(code, {}, data)

    return data['mp']

def HTMLDiagrams_hh(order, filename):
    #produce an HTML file for diagrams of order

    diagrams_hh.clear()

    #get combinations of diagrams
    connections = [[0] * nodalPairCount_hh(order)]
    nodePair = 0
    pairCount = nodalPairCount_hh(order)

    diagramCombinations = nodalPairConnectionsCombinations_hh(connections, nodePair, pairCount, order)

    #up arrow diagrams
    nDiagrams = upArrowCombinations_hh(diagramCombinations, order)

    #open file and write header
    f = open(filename, 'w')
    f.write('<!DOCTYPE html>\n<html lang="en">\n\t<body>\n\t\t<h6>order of diagrams ' + str(order) + '</h6>')

    for i, diagrams in enumerate(diagrams_hh):

        diagram, up = diagrams

        #down arrows
        down = downArrow_hh(diagram, up, pairCount)

        #arrow flow
        flows = connectionFlow_hh(up, down, order)

        #get rules
        rules = []
        rules.append(rules_hh(flows, order))

        #enumerate rules for diagrams
        for rule in rules:

            f.write('\n\t\t<p style=\'font-size : 12px;\'>&nbsp; diagram ' + str(diagram) + '</p>')
            f.write('\n\t\t<p style=\'font-size : 12px;\'>&nbsp; &nbsp; sub-diagram ' + str(up) + \
                                               '&#11165;' + ' ' + str(down) + '&#11167; </p>')

            h = str(rule['sign'][0]) + '+' + str(rule['sign'][1])
            f.write('\n\t\t<p style=\'margin-left: 240px;\'>\n\t\t\t<math>\n\t\t\t\t<mi mathsize=\'12px\'> &nbsp; &nbsp;(-1)<sup>' + h + '</sup></mi>')

            h = str(rule['powerTwo'])
            f.write('\n\t\t\t\t<mi mathsize=\'12px\'>&nbsp;(2)<sup>' + h + '</sup>&nbsp;</mi>')

            f.write('\n\t\t\t\t<mfrac>\n\t\t\t\t\t<mrow>')
            eris = rule['doubleBars'].split(',')
            for eri in eris:
                f.write('\n\t\t\t\t\t\t<mo><</mo><mi>' + eri[:2] +  '</mi><mo>||</mo><mi>' + eri[2:] + '</mi><mo>></mo>')
            f.write('\n\t\t\t\t\t</mrow>\n\t\t\t\t\t<mrow>')

            es = rule['orbitalEnergies']
            for e in es:
                s = ''
                for i in e:
                    if i.isalpha(): s += '<mi>&epsilon;<sub>' + i + '</sub></mi>'
                    if i in '+-': s += '<mo>' + i + '</mo>'
                f.write('\n\t\t\t\t\t\t(' + s + ')' )

            f.write('\n\t\t\t\t\t</mrow>\n\t\t\t\t</mfrac>\n\t\t\t</math>\n\t\t</p>')

    f.write('\n\t</body>\n</html>')
    f.close()

def mbptEvaluateMPn(molAtom, molBasis, molData, scf, mp = [2, 3, 4]):
    #evalute mbpt diagrams for mp2, mp3 and mp4
    from view import postSCF

    if not isinstance(mp, list): mp = [mp]

    mbpt = []
    for m in mp:
        mbpt.append(codeDiagrams_hh(molAtom, molBasis, molData, m))

    if mp == [2, 3, 4]: 
        mp2, mp3, mp4 = mbpt
        postSCF([mp2,mp3,mp4,scf,scf+mp2+mp3+mp4], 'mbpt')

    return mbpt
