from __future__ import division
from atom import weight, seperation, angle, oopAngle, dihedral, massCenter, bondMatrix, \
                 inertiaTensor, principalMoments, rotor, rotationalConstants, symbol,   \
                 nuclearRepulsion, getConstant, isBond, covalentRadius
from basis import electronCount, orbitalType, aufbau
from integral import iEri
from numpy import sqrt, zeros, linspace, sum
import math
import matplotlib.pyplot as py

def delimiter(f):
    #horizontal line
    f.write('<br><hr align="left" width="40%">')

def pre(name, type):

    # open HTML output file and write <head> and <style>
    f = open('harpy.html','w')
    f.write('<head>')
    f.write('\n\t<style>\n\t\ttable {\n\t\t\tfont-family: arial, sans-serif;\n\t\t\tborder-collapse: collapse;\n')
    f.write('\t\t\tmargin-left: 40px;\n\t\t\tfont-size:10px;\n\t\t}\n\t\ttd,th {\n\t\t\tborder: 1px solid #dddddd;\n')
    f.write('\t\t\ttext-align: center;\n\t\t\tpadding: 4px;\n\t\t}\n')
    f.write('\n\t\t#bond td{padding:10px}\n\t\t#bond tr{background-color:#FFFFFF}')

    f.write('\n\t</style>\n</head>\n')
    # start <body>
    f.write('<body style=\'background-color:white;font:10px\' onload=\'collapse();\'>\n')
    f.write('\t<p style=\'text-align:center;font-size:20px;text-decoration:underline;\'>')
    f.write('\t<span>HARPY ' + type.upper() + ' output for ' + name + '</span></p>')
    f.close()

def geometry(atoms):

    f = open('harpy.html','a')
    # start ATOMS section========================================================================================================
    f.write('\n\t<p style=\'text-align:center;font-weight:bold;border-style:solid;border-width:1px;\'>atoms</p>\n')
    f.write('\t<p style=\'font-size:10px;margin-left:40px;\' ><span>number of atoms is ' + str(len(atoms)) + '</span></p>')
    # list in input data
    f.write('\n\n\t<table>\n\t\t<caption>atomic input data</caption>')
    f.write('\n\t\t<tr><th></th><th>id</th><th>number</th><th>x</th><th>y</th><th>z</th><th>weight</th></tr>')
    for i in range(0, len(atoms)):
        f.write('\n\t\t<tr><td>' + str(i+1) + '</td><td>' + atoms[i].id + '</td><td>' + str(atoms[i].number) + '</td>',)
        for j in range(0,3):
            f.write('<td>' + str(atoms[i].center[j]) + '</td>',)
        f.write('<td>' + str(weight[atoms[i].number-1]) + '</td></tr>')
    f.write('\n\t</table>')

    # bond lengths
    if len(atoms) != 1:
        f.write('\n\t<br>\n\t<table>\n\t\t<caption>interatom seperations</caption>\n\t\t<tr><th>atoms</th><th>d (bohr)</th></tr>')
        for i in range(0, len(atoms)):
            for j in range(i+1, len(atoms)):
                # only seperation under 4 bohr
                if isBond(atoms, i, j):
                    d = seperation(atoms,i,j)
                    f.write('\n\t\t<tr><td>' + atoms[i].id + '-' + atoms[j].id + '</td><td>' + str(round(d,3)) + '</td></tr>')
        f.write('\n\t</table>')

    # angles
    if len(atoms) > 2:
        f.write('\n\t<br>\n\t<table>\n\t\t<caption>interatom angles</caption>\n\t\t<tr><th>atoms</th><th>angle</th></tr>')
        for i in range(0, len(atoms)):
            if sum(bondMatrix(atoms)[i,:]) <= 1: 
                continue
            for j in range(i+1, len(atoms)):
                if isBond(atoms,i,j):
                    for k in range(j+1, len(atoms)):
                        if k!=j and isBond(atoms,i,k):
                            f.write('\n\t\t<tr><td>' + atoms[j].id + '-' + atoms[i].id + '-' + atoms[k].id + \
                                '</td><td>' + str(round(angle(atoms,j,i,k),2)) + '</td></tr>')
        f.write('\n\t</table>')

    #out-of-plane angles
    if len(atoms) > 3:
        f.write('\n\t<br>\n\t<table>\n\t\t<caption>out-of-plane angles</caption>\n\t\t<tr><th>atoms</th><th>angle</th></tr>')
        for i in range(0, len(atoms)):
            for k in range(0, len(atoms)):
                for j in range(0, len(atoms)):
                    for l in range(0, j):
                        if (i!=j) and (i!=k) and (i!=l) and (j!=k) and (k!=l):
                            if isBond(atoms,i,k) and isBond(atoms,k,j) and isBond(atoms,k,l) :
                                f.write('\n\t\t<tr><td>' + atoms[i].id + '-' + atoms[j].id + '-' + atoms[k].id + '-' + atoms[l].id + \
                                '</td><td>' + str(round(oopAngle(atoms,i,j,k,l),2)) + '</td</tr>')
        f.write('\n\t</table>')

    #dihedral
        f.write('\n\t<br>\n\t<table>\n\t\t<caption>dihedral angles</caption>\n\t\t<tr><th>atoms</th><th>angle</th></tr>')
        for i in range(0, len(atoms)):
            for j in range(0, i):
                for k in range(0, j):
                    for l in range(0, k):
                        if (seperation(atoms,i,j) < 4) and (seperation(atoms,k,j) < 4) and (seperation(atoms,k,l) < 4):
                                f.write('\n\t\t<tr><td>' + atoms[i].id + '-' + atoms[j].id + '-' + atoms[k].id + '-' + atoms[l].id + \
                                '</td><td>' + str(round(dihedral(atoms,i,j,k,l),2)) + '</td</tr>')
        f.write('\n\t</table>')

    #center of mass
    f.write('\n\t<br>\n\t<table><caption>center of mass</caption>\n\t\t<tr><td>x</td><td>y</td><td>z</td></tr>\n\t\t<tr>',)
    com = massCenter(atoms)
    for i in range(0, 3):
        f.write('<td>' + str(round(com[i],3)) + '</td>',)
    f.write('</tr>\n\t</table>')

    #connection matrix
    f.write('\n\t<br>\n\t<table id=\'bond\'>\n\t\t<caption>inferred bonds</caption>\n\t\t<tr><th></th>')
    for i in range(0,len(atoms)):
        f.write('<th>' + atoms[i].id + '</th>',)
    f.write('</tr>')
    bonds = bondMatrix(atoms)
    for i in range(0, len(atoms)):
        f.write('<tr><td>' + atoms[i].id + '</td>')
        for j in range(0, len(atoms)):
            if bonds[i,j] == 1:
                f.write('<td>X</td>')
            else:
                f.write('<td></td>')
        f.write('</tr>')
    f.write('</tr>\n\t</table>')

    #inertia tensor
    tensor = inertiaTensor(atoms)

    f.write('\n\t<br>\n\t<table>\n\t\t<caption>inertia tensor</caption>\n\t\t<tr><th>x</th><th>y</th><th>z</th>')
    for i in range(0,3):
        f.write('\n\t\t<tr><td>' + str(round(tensor[i,0],6)) + '</td><td>'  + str(round(tensor[i,1],6)) + '</td><td>' \
                                 + str(round(tensor[i,2],6))+ '</td></tr>')
    f.write('\n\t</table>')
    
    #principal moments
    pm = principalMoments(atoms)

    f.write('\n\t<br>\n\t<table>\n\t\t<caption>principal moments (amu bohr<sup>2</sup>)</caption>')
    f.write('\n\t\t<tr><th>I<sub>a</sub></th><th>I<sub>b</sub></th><th>I<sub>c</sub></th>\n\t\t')
    f.write('\n\t\t<tr><td>' + str(round(pm[0],6)) + '</td><td>'  + str(round(pm[1],6)) + '</td><td>' \
                             + str(round(pm[2],6))+ '</td></tr>')
    f.write('\n\t</table>')
    
    #rotor type
    f.write('\n\t<br>\n\t<table><tr><th>rotor type</th><th>' + rotor(atoms) + '</th></tr></table>')

    #rotational constants
    rc = rotationalConstants(atoms)

    f.write('\n\t<br>\n\t<table width=240px>\n\t\t<caption>rotational constants (g cm<sup>-1</sup>)</caption>')
    f.write('\n\t\t<tr><th>A</th><th>B</th><th>C</th>\n\t\t')
    f.write('\n\t\t<tr><td>' + str(round(rc[0],4)) + '</td><td>'  +  str(round(rc[1],4)) + '</td><td>' \
                             + str(round(rc[2],4))+ '</td></tr>')
    f.write('\n\t</table>')

    #nuclear repulsion energy
    f.write('\n\t<br>\n\t<table><tr><th>nuclear repulsion energy</th><th>' + str(round(nuclearRepulsion(atoms),8)) \
                                                                           + '</th></tr></table>')
    f.close()

def orbitals(atoms, charge, multiplicity, name, bases, method):
    f = open('harpy.html','a')
    # start ORBITALS section========================================================================================================
    f.write('\n\t<br>\n\t<p style=\'text-align:center;font-weight:bold;border-style:solid;border-width:1px;\'>orbitals</p>\n')

    #number of electrons
    f.write('\n\t<br>\n\t<table><tr><th>number of electrons</th><th>' + str(electronCount(atoms,charge)) + '</th></tr></table>')

    #selected basis
    f.write('\n\t<br>\n\t<table><tr><th>basis set</th><th>' + name + '</th></tr></table>')

    #basis set information
    h = open('../basis/' + name + '.gbf', 'r')
    text=''
    while True:
        line = h.readline()
        if line.strip() == '' or line[:2] == '!-' :
            continue
        if line.strip() == '****':
            break
        text += line[1:].lstrip() + '<br>'
    h.close()
    if text[0] == 'a':
        text = 'C' + text 
    elif text[0] == 'p':
        text = 'S' + text 

    f.write('\n\t<br>\n\t<table><tr><th>' + text + '</th></tr></table>')

    if method == 'rhf':
        #orbital information
        f.write('\n\t<br>\n\t<table>\n\t\t<caption>orbital occupancy</caption>')
        f.write('\n\t\t<tr><td>number of basis functions</td><td>' + str(len(bases)) + '</td></tr>')
        f.write('\n\t\t<tr><td>number doubly occupied orbitals</td><td>' + str(int(electronCount(atoms,charge)/2)) + '</td></tr>')
        f.write('\n\t\t<tr><td>number of virtual orbitals</td><td>' + str(len(bases) - int(electronCount(atoms,charge)/2)) + '</td></tr>')
        f.write('\n\t</table>')
    else:
        alpha = electronCount(atoms,charge) - (multiplicity-1)
        f.write('\n\t<br>\n\t<table>\n\t\t<caption>orbital spin occupancy</caption>')
        f.write('\n\t\t<tr><td>number of alpha orbitals</td><td>' + str(alpha//2+(multiplicity-1)) + '</td></tr>')
        f.write('\n\t\t<tr><td>number of beta orbitals</td><td>' + str(alpha//2) + '</td></tr>')
        f.write('\n\t\t<tr><td>multiplicity</td><td>' + str(multiplicity) + '</td></tr>')
        f.write('\n\t</table>')

    #atomic orbitals
    f.write('\n\t<br>\n\t<table>\n\t\t<caption>atomic orbitals</caption>\n\t\t<td>')
    for i in range(0, len(bases)):
        type = orbitalType(bases[i].momentum)

        f.write('\n\t\t<td>' + str(i+1) + '</td><td>' + atoms[bases[i].atom].id + '</td><td>' + \
                symbol[atoms[bases[i].atom].number-1] + '</td><td>' + type[0] + '<sub>' + type[1:] + '</sub></td><td></td>',)
        if (i+1) % 4 == 0:
            f.write('</tr>\n\t\t<td>')
    f.write('\n\t\t</tr>\n\t</table>')

    #aufbau occupancy
    f.write('\n\t<br>\n\t<table>\n\t\t<caption>aufbau atomic occupancies</caption>')
    f.write('\n\t\t<tr><th></th><th>id</th><th>orbitals</th></tr>')
    for i in range(0 , len(atoms)):
        f.write('\n\t\t<tr><td>' + str(i+1) + '</td><td>' + atoms[i].id + '</td><td>' + aufbau(atoms[i]) + '</td></tr>')
    f.write('\n\t\t</tr>\n\t</table>')

    f.close()

def uhfOrbitals(alpha, beta, multiplicity):
    f = open('harpy.html','a')
    f.write('\n\t<br>\n\t<table>\n\t\t<caption>orbital spin occupancy</caption>')
    f.write('\n\t\t<tr><td>number of alpha orbitals</td><td>' + str(alpha) + '</td></tr>')
    f.write('\n\t\t<tr><td>number of beta orbitals</td><td>' + str(beta) + '</td></tr>')
    f.write('\n\t\t<tr><td>multiplicity</td><td>' + str(multiplicity) + '</td></tr>')
    f.write('\n\t</table>')
    f.close()

def preSCF(S,K,J,ERI,U, FO, D, IE, guess):
    f = open('harpy.html','a')
    # start PRE-SCF section========================================================================================================
    f.write('\n\t<br>\n\t<p style=\'text-align:center;font-weight:bold;border-style:solid;border-width:1px;\'>pre-scf</p>\n')
    f.write('\n\t<br>\n\t<table>\n\t\t<tr><th>initial fock guess</th><th>',)
    if guess == 'core':
        f.write('core hamiltonian</th>\n\t</table>')
    elif guess == 'gwh':
        f.write('generalised wolfsberg-helmholtz</th>\n\t</table>')

    showMatrix('overlap', S, f)
    showMatrix('kinetic', K, f)
    showMatrix('Coulomb', J, f)

    f.write('\n\t<style>\n\t\t#fixed-header {\n\t\t\twidth: 200px;\n\t\t\ttable-layout: fixed;\n\t\t\tborder-collapse: collapse;')
    f.write('\t\t}\n\t\t#fixed-header tbody{\n\t\t\tdisplay: block;\n\t\t\twidth: 100%;\n\t\t\toverflow: auto;\n\t\t\theight: 210px;')
    f.write('\t\t}\n\t\t#fixed-header thead tr {\n\t\t\tdisplay: block;\n\t\t}')
    f.write('\t\t#fixed-header th, #fixed-header td {\n\t\t\tpadding: 0px;\n\t\t\ttext-align: center;\n\t\t\twidth: 200px;\n\t\t}')
    f.write('\t</style>\n\t<br>')
    f.write('\t<table id=\'fixed-header\'>\n\t\t<caption>two electron repulsion integrals <sup>*</sup></caption>')
    f.write('\t\t<thead><tr><th width=\'20px\'>&#60; i j | k l &#62;</th><th></th></thead>\n\t\t<tbody>')

    n = S.shape[0]
    resume = True
    count = 0
    totalEri = n*(n+1)*(n*n + n + 2)/8
    for i in range(0 , n):
        for j in range(0 , n):
            for k in range(0 , n):
                for l in range(0 , n):
                    value = ERI[iEri(i,j,k,l)]
                    if abs(value) >= 1e-15:
                        f.write('\n\t\t\t<tr><td>&#60; ' + str(i+1) + ' ' + str(j+1) + ' | ' + str(k+1) + ' ' + str(l+1) + \
                                               ' &#62;</td><td>' + str(round(value, 6)) + '</td></tr>')
                        count += 1
                        if count == 30:
                            resume = False
                            break
                if not resume:
                    break
            if not resume:
                break
        if not resume:
            break
    f.write('\n\t\t</tbody>\n\t</table>')
    f.write('\n\t\t<p style=\'margin-left:40px;font-size:10px;\'><sup>*</sup>  non-zero integrals (total ' \
             + str(int(totalEri)) + ') first 30 shown</p>')

    showMatrix('S<sup>-&#189</sup>', U, f)
    showMatrix('orthogonal initial Fock', FO , f)
    showMatrix('initial density', D ,f)

    f.write('\n\t<br>\n\t<table><tr><td>initial SCF electronic energy</td><td>' + str(round(IE,8)) + '</td></tr></table>')

    f.close()

def SCF(e, de, dd, cycle, diis, iterations, convergence):
    f = open('harpy.html','a')
    # start SCF section========================================================================================================
    if cycle == 1:
        f.write('\n\t<br>\n\t<p style=\'text-align:center;font-weight:bold;border-style:solid;border-width:1px;\'>scf</p>\n')
        f.write('\n\t\t<table><tr><td>iteration limit</td><td>' + str(iterations) + '</td></tr><tr><td>convergence criterion</td><td>' \
                                   + str(convergence) + '</td></tr></table>')
        f.write('\n\t<br><p style=\'margin-left:40px;font-size:10px;\'>direct inversion of the iterative sub-space (diis) is <b>' \
                                                                       + diis + '</b></p>')
        f.write('\n\t<table>\n\t\t<tr><th>electronic energy (E)</th><th>&#916;(E)</th><th>rms(D)</th></tr>')
        f.write('\n\t\t<tr><td>' + "%.10f" % (de - e) + '</td><td>0</td><td>0</td></tr>')

    else:
        f.write('\n\t\t<tr><td>' + "%.10f" % (e) + '</td><td>' + "%.6e" % (de) + '</td><td>' + "%.6e" % (dd) + '</td></tr>')

def postSCF(data, type):
    f = open('harpy.html','a')
    # start post-SCF section========================================================================================================
    #close table from SCF section
    if type == 'uhf':
        f.write('\n\t</table>')
        f.write('\n\t<br><p style=\'margin-left:40px;font-size:10px;\'><b>SCF converged in </b>' + str(data[1]) + ' cycles</p>')
        f.write('\n\t<br>\n\t<table><tr><th>final scf electronic energy</th><th>' + "%.10f" % (data[2]) + '</th></tr>',)
        f.write('\n\t<br>\n\t<tr><th>final scf total energy</th><th>' + "%.10f" % (data[0]) + '</th></tr></table>')

    if type == 'eigen':
        f.write('\n\t</table>')
        f.write('\n\t<br><p style=\'margin-left:40px;font-size:10px;\'><b>SCF converged in </b>' + str(data[1]) + ' cycles</b></p>')

        f.write('\n\t<br>\n\t<p style=\'text-align:center;font-weight:bold;border-style:solid;border-width:1px;\'>post-scf</p>\n')
        f.write('\n\t<br>\n\t<table><tr><th>final scf total energy</th><th>' + "%.10f" % (data[0]) + '</th></tr></table>')
        f.write('\n\t<br>\n\t<table style=\'margin-left:54px;\'><caption>final orbital energies</caption>\n\t\t<tr>')

        for i in range(0, data[2].shape[0]):
            f.write('\n\t\t<th>' + str(i+1) + '</th>')
        f.write('</tr><tr>')
        for i in range(0, data[2].shape[0]):
            f.write('\n\t\t<th>' + "%.6f" % (data[3][i]) + '</th>')
        f.write('</tr>\n\t</table>')
        showMatrix('orbital coeffiecients', data[2], f, '%.6f')
        showMatrix('final density matrix', data[4], f, '%.6f')

    if type == 'uhf-post':
        f.write('\n\t<br>\n\t<table><caption>final orbital energies</caption>')
        f.write('\n\t\t<tr><td>alpha</td></tr>\n\t\t<tr>')
        for i in range(0, data[0].shape[0]):
            f.write('\n\t\t<th>' + str(i+1) + '</th>')
        f.write('</tr><tr>')
        for i in range(0, data[0].shape[0]):
            f.write('\n\t\t<th>' + "%.6f" % (data[0][i]) + '</th>')
        f.write('</tr>\n\t\t<tr><td>beta</td></tr>\n\t\t<tr>')
        for i in range(0, data[1].shape[0]):
            f.write('\n\t\t<th>' + str(i+1) + '</th>')
        f.write('</tr><tr>')
        for i in range(0, data[1].shape[0]):
            f.write('\n\t\t<th>' + "%.6f" % (data[1][i]) + '</th>')

        f.write('</tr>\n\t</table>')
        showMatrix('total density matrix',data[2],f, '%.6f')
        showMatrix('spin density matrix',data[3 ],f, '%.6f')

        f.write('\n\t<br><p style=\'margin-left:40px;font-size:10px;\'><b>&#60;S<sup>2</sup>&#62; is </b>' \
                 + str(round(data[4],6)))
        f.write('\n\t<br><p style=\'margin-left:40px;font-size:10px;\'><b>spin contamination is </b>' \
                 + "%.6f" % data[5])
        f.write('\n\t<br><p style=\'margin-left:40px;font-size:10px;\'><b>multiplicity is </b>' \
                 + "%.6f" % data[6])
      
    if type == 'uhf-mull':
        f.write('\n\t<br><br>\n\t<table><caption>mulliken population analysis</caption>')
        f.write('\n\t\t<tr><th>atom</th><th>orbital</th><th>&alpha;</th><th>&beta;</th><th>spin density</th></tr>')
        for i in range(data[0]):
            f.write('\n\t\t<tr><td>' + data[1][i][1] + '</td><td>' + data[1][i][0] + '</td><td>' + str(data[1][i][2]) + \
                   '</td><td>' + str(data[1][i+data[0]][2]) + '</td><td>' + str(round(data[5][i],6)) + '</td></tr>')
        f.write('\n\t\t<tr><td>&Sigma;</td><td></td><td>' + str(round(data[2][0])) + '</td><td>' + str(round(data[2][1])) + '</td><td>' + \
                                                            str(round(sum(data[5]),6)) + '</td></tr>')
        f.write('\n\t</table>')

        f.write('\n\t<br><br>\n\t<table><caption>reduced to atomic centers</caption>')
        f.write('\n\t\t<tr><th>atom</th><th>&alpha;</th><th>&beta;</th><th>&alpha;+&beta;</th><th>charge</th><th>spin density</th></tr>')
        for i in range(data[3].shape[1]):
            f.write('\n\t\t<tr><td>' + data[4][i][0] + '</td><td>' + str(round(data[3][0][i],6)) + '</td><td>' + str(data[3][1][i]) + \
                    '</td><td>' + str(round(data[3][0][i]+data[3][1][i],6)) + '</td><td>' + str(round(data[4][i][1] - \
                    data[3][0][i]-data[3][1][i],6)) + '</td><td>' + str(round(data[6][i],6)) +'</td></tr>')
        f.write('\n\t</table>')


    if type == 'mulliken':
        delimiter(f)
        f.write('\n\t<table><caption>charge analysis</caption>')
        f.write('\n\t<tr><td>orbital</td><td>atom</td><td>density</td></tr>')
        f.write('<tr><td colspan=\'3\' style=\'background-color:#dcdcdc\';>Mulliken</td></tr>')
        for i in range(0, len(data[1])):
            f.write('<tr><td>' + str(i+1) + '</td><td>' + data[3][data[1][i].atom].id + '</td><td>' \
                     + str(round(data[0][i], 4)) + '</td></tr>',)
        f.write('<tr><td colspan=\'3\'>reduced To atoms centers</td></tr>')
        for i in range(0, len(data[3])):
            f.write('<tr><td></td><td>' + data[3][i].id + '</td><td>' \
                     + str(round(data[2][i], 4)) + '</td></tr>',)
        f.write('<tr><td colspan=\'3\'>corrected for charge</td></tr>')
        for i in range(0, len(data[3])):
            f.write('<tr><td></td><td>' + data[3][i].id + '</td><td>' \
                     + str(round(-data[2][i] + data[3][i].number, 4)) + '</td></tr>',)

    if type == 'lowdin':
        f.write('<tr><td colspan=\'3\' style=\'background-color:#dcdcdc\';>' + type[0].upper() + type[1:] + '</td></tr>')
        for i in range(0, len(data[0])):
            f.write('<tr><td></td><td>' + data[1][i].id + '</td><td>' \
                     + str(round(data[0][i], 4)) + '</td></tr>',)
        f.write('<tr><td colspan=\'3\'>corrected for charge</td></tr>')
        for i in range(0, len(data[0])):
            f.write('<tr><td></td><td>' + data[1][i].id + '</td><td>' \
                     + str(round(-data[0][i] + data[1][i].number, 4)) + '</td></tr>',)

        f.write('\n\t</table>')

    if type == 'bonds':
        delimiter(f)
        f.write('\n\t<table><caption>Mayer bond orders and valence</caption>')
        f.write('\n\t<tr><th></th>',)
        for i in range(1,len(data[0])):
            f.write('<th>' + data[0][i].id + '</th>',)
        f.write('<th>valence</th></tr>')
        for i in range(len(data[0])):
            f.write('\n\t\t<tr><td>' + data[0][i].id + '</td>',)
            if i != 0: f.write('<td colspan=\'' + str(i) + '\'></td>',)
            for j in range(1,len(data[0])):
                if j > i: 
                    f.write('<td>' + str(data[1][i,j]) + '</td>',)
            f.write('<td>' + str(data[2][i]) + '</tr>')

        f.write('\n\t</table>')

    if type == 'energy':
        delimiter(f)
        f.write('\n\t<table><caption>energy partition</caption>')
        f.write('\n\t<tr><th>description</th><th>components</th><th>energy</th></tr>')
        lookup = ['k','n', 'j','1','2','t','p','tk','v','b','e~']
        for i in range(0, len(lookup)):
            f.write('\t\t<tr><td>' + data[lookup[i]][1] + '</td><td>' +  data[lookup[i]][2] + '</td><td>' + \
                    str(round( data[lookup[i]][0], 6)) + '</td></tr>')
        f.write('\n\t</table>')

    if type == 'dipole':
        delimiter(f)
        f.write('\n\t<table><caption>dipoles</caption>')
        f.write('\n\t<tr><th></th><th>x</th><th>y</th><th>z</th><th>(au)</th><th>(debye)</th></tr>')
        f.write('\n\t\t<tr><td></td><td>' + str(round(data[0], 4)) + '</td><td>' + \
                 str(round(data[1], 4)) + '</td><td>' + str(round(data[2], 4)) + '</td><td>' + \
                 str(round( sqrt(data[3]), 4)) + '</td><td>' + str(round( sqrt(data[3])* getConstant('au->debye'), 4) ) + '</td></tr>')
        f.write('\n\t</table>')

    if type == 'mp2-dipole':
        f.write('\n\t<table><caption>mp2 dipoles</caption>')
        f.write('\n\t<tr><th></th><th>x</th><th>y</th><th>z</th><th>(au)</th><th>(debye)</th></tr>')
        f.write('\n\t\t<tr><td></td><td>' + str(round(data[0], 4)) + '</td><td>' + \
                 str(round(data[1], 4)) + '</td><td>' + str(round(data[2], 4)) + '</td><td>' + \
                 str(round( sqrt(data[3]), 4)) + '</td><td>' + str(round( sqrt(data[3])* getConstant('au->debye'), 4) ) + '</td></tr>')
        f.write('\n\t</table>')        

    if type == 'quad':
        delimiter(f)
        f.write('\n\t<table><caption>quadrupoles (debye &#8491)</caption>')
        f.write('\n\t<tr><th>x<sup>2</sup></th><th>y<sup>2</sup></th><th>z<sup>2</sup></th>' + \
                                 '<th>xy</th><th>yz</th><th>zx</th></tr>')
        f.write('\n\t\t<tr><td>' + str(round(data[0], 4)) + '</td><td>' + str(round(data[1], 4)) + '</td><td>' + str(round(data[2], 4)) + '</td>' + \
                          '<td>' + str(round(data[3], 4)) + '</td><td>' + str(round(data[4], 4)) + '</td><td>' + str(round(data[5], 4)) + '</td></tr>')
        f.write('\n\t\t<tr><td colspan=\'6\'>traceless</td></tr>')

        f.write('\n\t\t<tr><td>' + str(round(data[6], 4)) + '</td><td>' + str(round(data[7], 4)) + '</td><td>' + \
            str(round(data[8], 4)) + '</td>' + '<td>' + str(round(data[3], 4)) + '</td><td>' + str(round(data[4], 4)) + '</td><td>' + \
            str(round(data[5], 4)) + '</td></tr>')
        f.write('\n\t\t<tr><td colspan=\'6\'>eigenvalues</td></tr>')
        f.write('\n\t\t<tr><td>' + str(round(data[9], 4)) + '</td><td>' + str(round(data[10], 4)) + '</td><td>' + \
            str(round(data[11], 4)) +'</td></tr>')

        f.write('\n\t\t<tr><td colspan=\'5\'>trace is close to zero</td><td>' + str((data[9]+data[10]+data[11]) < 1e-8) + '</td></tr>')
        f.write('\n\t\t<tr><td colspan=\'4\'>amplitude</td><td colspan=\'2\'>' +  str(round(data[12], 4))  + '</td></tr>')
        f.write('\n\t\t<tr><td colspan=\'4\'>asymmetry</td><td colspan=\'2\'>' +  str(round(data[13], 4))  + '</td></tr>')

        f.write('\n\t</table>')

    if type == 'mp':
        delimiter(f)
        f.write('\n\t<table><caption>moller-plesset</caption>')
        f.write('\n\t<tr><td>{mp2} parallel spin</td><td>' + "%.8f" % round(data[0], 8) + '</td></tr>')
        f.write('\n\t<tr><td>{mp2} anti-parallel spin</td><td>' + "%.8f" % round(data[1], 8) + '</td></tr>')
        f.write('\n\t<tr><td>{mp2} total</td><td>' + "%.8f" % round(data[0]+data[1], 8)+ '</td></tr>')

        f.write('<tr><td colspan=\'2\'></td></tr>')
        f.write('\n\t<tr><td>{mp3} total</td><td>' + "%.8f" % round(data[2], 8) + '</td></tr>')
        f.write('<tr><td colspan=\'2\'></td></tr>')
        f.write('\n\t<tr><td>{mp2+mp3} total</td><td>' + "%.8f" % round(data[0]+data[1]+data[2], 8) + '</td></tr>')
        f.write('<tr><td colspan=\'2\'></td></tr>')
        f.write('\n\t<tr><td>{total energy + mp correction} total</td><td>' + "%.8f" % \
                              round(data[0]+data[1]+data[2]+data[3], 8) + '</td></tr>')
        f.write('\n\t</table>')

    if type == 'omp':
        delimiter(f)
        f.write('\n\t<table><caption>orbital optimised mp2</caption>')
        f.write('\n\t\t<tr><th>cycle</th><th>total energy</th><th>&#916;(E)</th></tr>')
        f.write('\n\t\t<tr><td>' + str(data[0]) + '</td><td>' + "%15.8f" % data[1] + '</td><td>' + "%2.5e" % data[2] + '</td></tr>')
        f.write('\n\t</table>')

    if type == 'mbpt':
        delimiter(f)
        f.write('\n\t<table><caption>many-body perturbation diagrams</caption>')
        f.write('\n\t\t<tr><td>mp2 correlation</td><td>' + "%10.8f" % data[0] + '</td></tr>')
        f.write('\n\t\t<tr><td>mp3 correlation</td><td>' + "%10.8f" % data[1] + '</td></tr>')
        f.write('\n\t\t<tr><td>mp4 correlation</td><td>' + "%10.8f" % data[2] + '</td></tr>')
        f.write('\n\t\t<tr><td>SCF energy</td><td>' + "%10.8f" % data[3] + '</td></tr>')
        f.write('\n\t\t<tr><td>Total corrected energy</td><td>' + "%10.8f" % data[4] + '</td></tr>')
        f.write('\n\t</table>')

    if type == 'mplp':
        delimiter(f)
        f.write('\n\t<table><caption>Laplace transform generated mp2</caption>')
        f.write('\n\t\t<tr><th>method</th><th1<energy (au)</th></tr>')
        f.write('\n\t\t<tr><td>reference scf energy</td><td>' + "%10.8f" % data[1] + '</td></tr>')
        f.write('\n\t\t<tr><td>parallel spin correlation</td><td>' + "%10.8f" % (data[0][1]) + '</td></tr>')
        f.write('\n\t\t<tr><td>anti-parallel spin correlation</td><td>' + "%10.8f" % (data[0][0]) + '</td></tr>')
        f.write('\n\t\t<tr><td>total spin correlation</td><td>' + "%10.8f" % (data[0][0]+data[0][1]) + '</td></tr>')
        f.write('\n\t\t<tr><td>corrected scf energy</td><td>' + "%10.8f" % (data[0][0]+data[0][1]+data[1]) + '</td></tr>')
        f.write('\n\t\t<tr><td>total spin-component scaled</td><td>' + "%10.8f" % (data[0][0]*1.2+data[0][1]/3) + '</td></tr>')
        f.write('\n\t\t<tr><td>corrected spin-component scaled</td><td>' + "%10.8f" % (data[0][0]*1.2+data[0][1]/3+data[1]) + '</td></tr>')
        f.write('\n\t</table>')

    if (type == 'ci') or (type == 'rpa'):
        if type == 'ci':
            title = 'configuration interaction' 
        if type == 'rpa':
            title = 'random phase approximation'
        delimiter(f)
        f.write('\n\t<table><caption>' + title + '<sup> *</sup></caption>\n\t\t<tr>')
        for i in range(0, len(data)):
            f.write('\n\t\t<td>' + "%.6f" % round(data[i][0], 6) + ' (' + data[i][1] + ')</td>')
            if ((i % 6) == 5) and (i != len(data)):
                f.write('\n\t\t</tr>\n\t\t<tr>')
        f.write('\n\t\t</tr>\n\t</table>')
        f.write('\n\t\t<p style=\'margin-left:40px;font-size:10px;\'><sup>*</sup> s -singlet, t -triplet</p>')

    if type[:4] == 'cisa':
        if type[4] == 's':
            degeneracy = 'singles'
        elif type[4] == 't':
            degeneracy = 'triplets'

        delimiter(f)
        f.write('\n\t<table><caption>configuration interaction spin-adapted ' + degeneracy + \
                '</caption>\n\t\t<tr>')
        for i in range(0, len(data)):
            f.write('\n\t\t<td>' + "%.6f" % round(data[i], 6) + '</td>')
            if ((i % 6) == 5) and (i != len(data)):
                f.write('\n\t\t</tr>\n\t\t<tr>')
        f.write('\n\t\t</tr>\n\t</table>')

    if type == 'fci':
        delimiter(f)
        f.write('\n\t<table><caption>full configuration interaction</caption>')
        f.write('\n\t\t<tr><td>number of electrons</td><td>' + str(data[0]) + '</tr>')
        f.write('\n\t\t<tr><td>number of spin orbitals</td><td>' + str(data[1]) + '</tr>')
        f.write('\n\t\t<tr><td>number of determinants</td><td>' + str(data[2]) + '</tr>')
        f.write('\n\t\t<tr><td>SCF energy</td><td>' + "%10.6f" % data[3] + '</tr>')
        f.write('\n\t\t<tr><td>FCI energy</td><td>' + "%10.6f" % (data[4] + data[5]) + '</tr>')
        f.write('\n\t\t<tr><td>FCI correction</td><td>' + "%10.6f" % (data[4] + data[5] - data[3]) + '</tr>')
        f.write('\n\t</table>')

    if type == 'cisd':
        delimiter(f)
        f.write('\n\t<table><caption>ci singles and doubles</caption>')
        f.write('\n\t\t<tr><td>number of electrons</td><td>' + str(data[0]) + '</tr>')
        f.write('\n\t\t<tr><td>number of spin orbitals</td><td>' + str(data[1]) + '</tr>')
        f.write('\n\t\t<tr><td>number of determinants</td><td>' + str(data[2]) + '</tr>')
        f.write('\n\t\t<tr><td>SCF energy</td><td>' + "%10.6f" % data[3] + '</tr>')
        f.write('\n\t\t<tr><td>CISD energy</td><td>' + "%10.6f" % (data[4] + data[5]) + '</tr>')
        f.write('\n\t\t<tr><td>CISD correction</td><td>' + "%10.6f" % (data[4] + data[5] - data[3]) + '</tr>')
        f.write('\n\t</table>')
        
    if type == 'cis':
        delimiter(f)
        f.write('\n\t<table><caption>ci singles (slater determinants)</caption>')
        f.write('\n\t\t<tr><td>number of electrons</td><td>' + str(data[0]) + '</tr>')
        f.write('\n\t\t<tr><td>number of spin orbitals</td><td>' + str(data[1]) + '</tr>')
        f.write('\n\t\t<tr><td>number of determinants</td><td>' + str(data[2]) + '</tr>')
        for i in range(0, len(data[3])):
            f.write('\n\t\t<td>' + "%.6f" % round(data[3][i]*getConstant('hartree->eV'), 6) + '</td>')
            if ((i % 6) == 5) and (i != len(data)):
                f.write('\n\t\t</tr>\n\t\t<tr>')
        f.write('\n\t\t</tr>\n\t</table>')

    if type == 'bd':
        if len(data) == 0:
            f.write('\n\t\t<p style=\'margin-left:40px;font-size:10px;\'>block Davidson failed</p>')
        else:
            f.write('\n\t<br>\n\t<table><caption>block Davidson</caption>\n\t\t<tr>')
            for i in range(0, len(data)):
                f.write('\n\t\t<td>' + "%.6f" % round(data[i], 6) +  '</td>')
            f.write('\n\t\t</tr>\n\t</table>')

    if type == 'ju':
        state = -1
        count = 0
        f.write('\n\t<br>\n\t<table><caption>CIS significant excitations</caption>\n\t\t<tr>')
        f.write('\n\t\t<tr><th>state</th><th>energy</th><th>(%) excitation</tr><tr>')
        for i in range(0, len(data)):
            if count == 10:
                break
            if data[i][0] != state:
                f.write('\n\t\t</tr><tr><td>' + str(data[i][0]) + '</td><td>' + "%.6f" % round(data[i][1], 6) + '</td><td>(' + str(data[i][2]) + \
                         ')  ' + data[i][3] + '</th>')
                state = data[i][0]
                count += 1
            else:
                f.write('\n\t\t<td>(' + str(data[i][2]) + ')  ' + data[i][3] + '</td>')
        f.write('\n\t\t</tr>\n\t</tr></table>')

        f.write('\n\t\t<p style=\'margin-left:40px;font-size:10px;\'>contributions over 10% of 10 lowest energy levels</p>')


    if 'diis' in type:
        if data[3] == 2:
            delimiter(f)
            f.write('<p style=\'margin-left:40px;font-size:10px;\'>direct inversion of the iterative sub-space (diis) is <b>' \
                                                                       + data[4] + '</b></p>')
            f.write('\n\t\n\t\t<table>\n\t\t<caption>coupled-cluster interations</caption>  \
                     \n\t\t<tr><th>ccsd energy</th><th>&#916;(E)</th><th>&#916;rms(amplitudes)</th></tr>')
            f.write('\n\t\t<tr><td>' + "%.10f" % (data[0] - data[1]) + '</td><td>0</td><td>0</td></tr>')
        else:
            f.write('\n\t\t<tr><td>' + "%.10f" % data[0] + '</td><td>' + "%.6e" % data[1] + '</td><td>' + "%.6e" % data[2] + '</td></tr>')

    if type == 'ccsd(t)':
        f.write('\n\t</table>')
        f.write('\n\t<p style=\'margin-left:40px;font-size:10px;\'><b>ccsd converged in </b>' + str(data[0]) + ' cycles</b></p>')
        f.write('\n\t<table><caption>coupled cluster</caption>')
        f.write('\n\t\t<tr><td>mp2 correction</td><td>' + "%.8f" % round(data[3], 8) + '</td></tr>')
        f.write('<tr><td colspan=\'2\'></td></tr>')     
        f.write('\n\t\t<tr><td>singles and doubles CCSD</td><td>' + "%.8f" % round(data[1], 8) + '</td></tr>')
        f.write('\n\t\t<tr><td>perturbative triples CCSD(T)</td><td>' + "%.8f" % round(data[2], 8) + '</td></tr>')
        f.write('\n\t\t<tr><td>total ccsd correction</td><td>' + "%.8f" % round(data[2]+data[1], 8) + '</td></tr>')
        f.write('<tr><td colspan=\'2\'></td></tr>')     
        f.write('\n\t\t<tr><td>total energy ccsd corrected</td><td>' + "%.8f" % round(data[4]+data[1]+data[2], 8) + '</td></tr>')
        f.write('\n\t</table>')

    if type == 'lccd':
        f.write('\n\t</table>')
        f.write('\n\t<p style=\'margin-left:40px;font-size:10px;\'><b>lccd converged in </b>' + str(data[0]) + ' cycles</b></p>')
        f.write('\n\t<table><caption>linear coupled cluster doubles <sup>*</sup></caption>')
        f.write('\n\t\t<tr><td>lccd correction</td><td>' + "%.8f" % round(data[1], 8) + '</td></tr>')
        f.write('\n\t\t<tr><td>corrected energy</td><td>' + "%.8f" % round(data[2]+data[1], 8) + '</td></tr>')
        f.write('\n\t</table>')
        f.write('\n\t\t<p style=\'margin-left:40px;font-size:10px;\'><sup>*</sup>coupled electron pair approximation (zero)</p>')

    if type == 'ccd':
        f.write('\n\t</table>')
        f.write('\n\t<p style=\'margin-left:40px;font-size:10px;\'><b>ccd converged in </b>' + str(data[0]) + ' cycles</b></p>')
        f.write('\n\t<table><caption>coupled cluster doubles <sup></sup></caption>')
        f.write('\n\t\t<tr><td>ccd correction</td><td>' + "%.8f" % round(data[1], 8) + '</td></tr>')
        f.write('\n\t\t<tr><td>corrected energy</td><td>' + "%.8f" % round(data[2]+data[1], 8) + '</td></tr>')
        f.write('\n\t</table>')

    if type == 'cc2':
        f.write('\n\t</table>')
        f.write('\n\t<p style=\'margin-left:40px;font-size:10px;\'><b>cc2 converged in </b>' + str(data[0]) + ' cycles</b></p>')
        f.write('\n\t<table><caption>coupled cluster CC2 <sup></sup></caption>')
        f.write('\n\t\t<tr><td>cc2 correction</td><td>' + "%.8f" % round(data[1], 8) + '</td></tr>')
        f.write('\n\t\t<tr><td>corrected energy</td><td>' + "%.8f" % round(data[3]+data[1], 8) + '</td></tr>')
        f.write('\n\t</table>')

    if type == 'lccsd':
        f.write('\n\t</table>')
        f.write('\n\t<p style=\'margin-left:40px;font-size:10px;\'><b>lccsd converged in </b>' + str(data[0]) + ' cycles</b></p>')
        f.write('\n\t<table><caption>linear coupled-cluster singles and doubles<sup></sup></caption>')
        f.write('\n\t\t<tr><td>lccsd correction</td><td>' + "%.8f" % round(data[1], 8) + '</td></tr>')
        f.write('\n\t\t<tr><td>corrected energy</td><td>' + "%.8f" % round(data[2]+data[1], 8) + '</td></tr>')
        f.write('\n\t</table>')

    if type == '+c':
        f.write('\n\t<br><table>')
        if data[0] in ['ccd','ccsd','cc2','lccd','lccsd']:
            f.write('\n\t\t<tr><th>method</th><th>correction</th><th>mp2</th><th>total electronic</th><th>total</th></tr>')
        elif data[0] in ['ccsd(t)']:
            f.write('\n\t\t<tr><th>method</th><th>sd correction</th><th>t perturbation</th><th>mp2</th><th>total electronic</th><th>total</th></tr>')
        else:
            f.write('\n\t\t<tr><th>method</th><th>correction</th><th>total electronic</th><th>total</th></tr>')

        eTotal = data[1]['eHF'] + data[1]['cc']
        f.write('\n\t\t<tr><td>' + 'fast ' + data[0] + '</td><td>' + "%.8f" % round(data[1]['cc'], 8) + '</td>')
        if data[0] in ['ccsd(t)']:
            f.write('\n\t\t<td>' + "%.8f" % round(data[1]['pt'],8) + '</td>')
            eTotal += data[1]['pt']
        if data[0] != 'lambda': f.write('\n\t\t<td>' + "%.8f" % round(data[1]['mp2'], 8) + '</td>')
        f.write('<td>' + "%.8f" % round(eTotal, 8) + \
                '</td><td>' + "%.8f" % round(eTotal+data[1]['nuclear'], 8) + '</td></tr>')
        f.write('\n\t</table>')

    if type == 'fa':
        delimiter(f)
        f.write('\n\t<table><caption>molecular forces (E<sub>h</sub>a<sub>0</sub><sup>-1)</sup></caption>')
        f.write('\n\t\t<tr><th>atom</th><th>x</th><th>y</th><th>z</th></tr>')
        f.write('\n\t\t<tr><td colspan=\'4\'>analytic</td></tr>')
        for i in range(0, len(data[1])):
            f.write('\n\t\t<tr><td>' + data[1][i].id + '</td><td>' + "%.6f" % data[0][i,0] + '</td><td>' + "%.6f" % data[0][i,1] + '</td><td>' \
                                     + "%.6f" % data[0][i,2] + '</td></tr>')
        f.write('\n\t</table>')
      
    if type == 'fn':
        f.write('\n\t<table><caption>molecular forces (E<sub>h</sub>a<sub>0</sub><sup>-1)</sup></caption>')
        f.write('\n\t\t<tr><td colspan=\'4\'>numeric</td></tr>')
        for i in range(0, len(data[1])):
            f.write('\n\t\t<tr><td>' + data[1][i].id + '</td><td>' + "%.6f" % data[0][i,0] + '</td><td>' + "%.6f" % data[0][i,1] + '</td><td>' \
                                     + "%.6f" % data[0][i,2] + '</td></tr>')
        f.write('\n\t</table>')

    if type in ['ep', 'eps', 'ep3']:
        delimiter(f)
        f.write('\n\t<table><caption>electron propogator - Koopman\'s theorem</caption>')

    if type == 'ep':
        f.write('\n\t\t<tr><th>HOMO - n</th><th>Koopman (eV)</th><th>EP2 (eV)</th></tr>')
        f.write('\n\t\t<tr><td colspan=\'3\'><b>ep2 spatial</b></td></tr>')
        koopman = data[2] * data[3]

        for orbital in range(0, len(data[1])):
            kpOrbital = data[0] + orbital + 1
            f.write('\n\t\t\t<tr><td>' + "% 4d" % (kpOrbital - data[4] + 1) + '</td><td>' + "%.4f" % koopman[kpOrbital] + \
                     '</td><td>' + "%.4f" % data[1][orbital] + '</td></tr>')
    if type == 'eps':
        f.write('\n\t\t<tr><th>HOMO - n</th><th>Koopman (eV)</th><th>EP2 (eV)</th></tr>')
        f.write('\n\t<tr><td colspan=\'3\'><b>ep2 spin</b></td></tr>')
        koopman = zeros(len(data[0]))
        for i in range(0, len(data[0])):
            koopman[i] = data[1][i*2] * getConstant('hartree->eV')
        for orbital in range(0, len(data[0])):
            f.write('\n\t\t\t<tr><td>' + "% 4d" % (len(data[0])-orbital-1) + '</td><td>' + "%.4f" % koopman[orbital] + \
                     '</td><td>' + "%.4f" % data[0][orbital] + '</td></tr>')
    if type == 'ep3':
        f.write('\n\t\t<tr><th>HOMO - n</th><th>Koopman (eV)</th><th>EP3 (eV)</th></tr>')
        f.write('\n\t<tr><td colspan=\'3\'><b>ep3 spin</b></td></tr>')
        koopman = zeros(len(data[0]))
        for i in range(0, len(data[0])):
            koopman[i] = data[1][i*2] * getConstant('hartree->eV')
        for orbital in range(0, len(data[0])):
            f.write('\n\t\t\t<tr><td>' + "% 4d" % (len(data[0])-orbital-1) + '</td><td>' + "%.4f" % koopman[orbital] + \
                     '</td><td>' + "%.4f" % data[0][orbital] + '</td></tr>')
        f.write('\n\t</table>')
    if type in ['ep', 'eps', 'ep3']:
        f.write('\n\t</table><br>')
        
    if type == 'gfa':
        if data[0] == -1:
            delimiter(f)
            f.write('\n\t<table><caption>Approximate Greens function correction to Koopmans theorem (eV)</caption>')
            f.write('\n\t\t<tr><th>HOMO - n</th><th>Koopman</th><th>orbital relaxation</th><th>pair relaxation</th>' + \
                '<th>pair removal</th><th>correction</th><th>KP + correction</th></tr>')
        f.write('\n\t\t<tr><td>' + "% 2d" % data[1] + '</td><td>' + "%.4f" % data[2] + '</td><td>' + "%.4f" % data[3] + \
                     '</td><td>' + "%.4f" % data[4] + '</td><td>' + "%.4f" % data[5] + '</td><td>' + "%.4f" % data[6] \
                   + '</td><td>' + "%.4f" % data[7] )
        if data[0] == 1:
            f.write('\n\t</table>')

    if type == 'po':
        delimiter(f)
        f.write('\n\t<table><caption>polarizability (&#945) (coupled-perturbed SCF)</caption>')
        f.write('\n\t\t<tr><td colspan=\'3\'><b>principal polarizabilities</b></td></tr>')
        f.write('\n\t\t<tr><td>' + "%.4f" % data[0][0] + '</td><td>' + "%.4f" % data[0][1] + '</td><td>' \
                                 + "%.4f" % data[0][2] + '</td></tr>')
        f.write('\n\t\t<tr><td colspan=\'2\'><b>isotropic</b></td><td>' + "%.4f" % data[1] + '</td></tr>')
        f.write('\n\t</table>')

    if type == 'hyper':
        delimiter(f)
        f.write('\n\t<table><caption>static hyperpolarizability (&#946) (au)</caption>')
        f.write('\n\t\t<tr><th>x</th><th>y</th><th>z</th></tr>')
        f.write('\n\t\t<tr><td>' + "%.4f" % data[0][0] + '</td><td>'+ "%.4f" % data[0][1] + '</td><td>'+ "%.4f" % data[0][2] + '</td></tr>')
        f.write('\n\t\t<tr><td colspan=\'2\'>amplitude</td><td>' + "%.4f" % data[1] + '</td></tr>')
        f.write('\n\t\t<tr><td colspan=\'2\'>&#8741<sub>z</sub></td><td>' + "%.4f" % data[2] + '</td></tr>')
        f.write('\n\t\t<tr><td colspan=\'2\'>&#8869<sub>z</sub></td><td>' + "%.4f" % data[3] + '</td></tr>')

        f.write('\n\t</table>')

    if type == 'resp':
        delimiter(f)
        if data[5] == '1':
            f.write('<p style=\'text-align:center;font-size:20px;\'>')
            f.write('\t<span>restrained electrostatic potential charges</span></p>')

            f.write('<br>\n\t<table>')
            f.write('\n\t\t<tr><td>molecular basis</td><td>' + data[0]['basis']+ '</td>')
            f.write('\n\t\t<td>molecular charge</td><td>' + str(data[0]['charge']) + '</td></tr>')
            f.write('\n\t\t<tr><td colspan=\'4\'></td></tr>')
            if data[1]['sphere'] == 'con' : s = 'Connolly'
            else: s = 'Fibonacci'
            f.write('\n\t\t<tr><td>spherical distribution model</td><td>' + s + '</td>')
            if data[1]['points'][0] == 'density':
                f.write('\n\t\t<td>surface density</td><td>' + str(data[1]['points'][1]) + '</td></tr>')
            else: f.write('\n\t\t<td>specified points</td><td>' + str(data[1]['points'][1]) + '</td></tr>')
            f.write('\n\t\t<tr><td>number of shells</td><td>' + str(data[1]['shell'][0])  + '</td>')
            if data[1]['shell'][0] == 1:
                f.write('\n\t\t<td>VdW scaling factor</td><td>' + str(data[1]['shell'][2]) + '</td></tr>')
                f.write('\n\t\t<tr><td colspan=\'4\'></td></tr>')
            else:
                f.write('\n\t\t<td>shell increment</td><td>' + str(data[1]['shell'][1]) + '</td>')
                f.write('\n\t\t<td>base shell</td><td>' + str(data[1]['shell'][2]) + '</td></tr>')
                f.write('\n\t\t<tr><td colspan=\'6\'></td></tr>')

            f.write('\n\t\t<tr><td>effective evaluation points</td><td>' + str(data[4][0]) + '</td><td>restrained iterations</td><td>' + \
                    str(data[4][1]) + '</td><td>&#916rms (resp - classical)</td><td>' + str(round(data[4][2],4)) + '</td></tr>')
            f.write('\n\t\t<tr><td colspan=\'6\'></td></tr>')

            for i in range(len(data[1]['constrain'])):
                s = data[1]['constrain'][i]
                t = ''
                for j in range(len(s[1])): t += str(abs(s[1][j])) + ' '
                if s[1][0] < 0 :
                    f.write('\n\t\t<tr><td>constrained charges on atoms ' + t + ' are equal</td></tr>')
                else:
                    f.write('\n\t\t<tr><td>constrained charges on atoms ' + t + ' sum to ' + str(round(s[0],4)) + '</td></tr>')

            f.write('\n\t\t<tr><td colspan=\'8\'></td></tr>')
            s = data[1]['restrain']
            f.write('\n\t\t<tr><td><b>A</b> scaling</td><td>' + str(s['a']) + '</td><td><b>b</b> scaling</td><td>' + str(s['b']) + '</td>')
            f.write('\n\t\t<td>convergence tolerance</td><td>' + str(s['tol']) + '</td><td>iteration limit</td><td>' + str(s['cycles']) + '</td></tr>')
            if s['h'] : f.write('\n\t\t<td>hydrogens are restrained</td></tr>')
            else: f.write('\n\t\t<td>hydrogens are <b>not</b> restrained</td></tr>')

            f.write('\n\t\t<tr><td colspan=\'1\'></td></tr>')
            f.write('\n\t\t<tr><td>constrained electrostatic potential charges</td></tr>')
            f.write('\n\t\t<tr>')
            for i in range(len(data[2])):
                f.write('\n\t\t<td>' + str(round(data[2][i],4)) + '</td>')
            f.write('\n\t\t</tr>')

            t = len(data[2])
            f.write('\n\t\t<tr><td colspan=' + str(t) + '></td></tr>')
            f.write('\n\t\t<tr><td>restrained electrostatic potential charges</td></tr>')
            f.write('\n\t\t<tr>')
            for i in range(len(data[3])):
                f.write('\n\t\t<td>' + str(round(data[3][i],4)) + '</td>')
            f.write('\n\t\t</tr>')
        if data[5] == '2':
            t = len(data[2])
            f.write('\n\t\t<tr><td colspan=' + str(t) + '><b>stage two constraints</b></td></tr>')
            f.write('\n\t\t<tr><td>restrained iterations</td><td>' + str(data[4][0]) + '</td><td>&#916rms (resp - classical)</td><td>' + \
                           str(round(data[4][1],4)) + '</td></tr>')
            for i in range(len(data[1]['constrain'])):
                s = data[1]['constrain'][i]
                t = ''
                for j in range(len(s[1])): t += str(abs(s[1][j])) + ' '
                if s[1][0] < 0 :
                    f.write('\n\t\t<tr><td>constrained charges on atoms ' + t + ' are equal</td></tr>')
                else:
                    f.write('\n\t\t<tr><td>constrained charges on atoms ' + t + ' sum to ' + str(round(s[0],5)) + '</td></tr>')
            f.write('\n\t\t<tr><td>constrained electrostatic potential charges</td></tr>')
            f.write('\n\t\t<tr>')
            for i in range(len(data[2])):
                f.write('\n\t\t<td>' + str(round(data[2][i],4)) + '</td>')
            f.write('\n\t\t</tr>')

            t = len(data[2])
            f.write('\n\t\t<tr><td colspan=' + str(t) + '></td></tr>')
            f.write('\n\t\t<tr><td>restrained electrostatic potential charges</td></tr>')
            f.write('\n\t\t<tr>')
            for i in range(len(data[3])):
                f.write('\n\t\t<td>' + str(round(data[3][i],4)) + '</td>')
            f.write('\n\t\t</tr>')

    if type == 'eom':
        delimiter(f)
        f.write('\n\t<p><b>equation of motion</b></p>')
        f.write('\n\t<p style=\'margin-left:20px;font-size:10px;\'>coupled-cluster singles and doubles calculation<br>')
        f.write('\n\t<br><table><tr><td>scf energy&nbsp;&nbsp;&nbsp;</td><td>' + str(round(data[0][0],10)) \
                                                                           + '</td></tr>')
        f.write('\n\t<tr><td>ccsd energy</td><td>' + str(round(data[0][1],10)) \
                                                                           + '</td></tr>')
        f.write('\n\t<tr><td>total energy</td><td>' + str(round(data[0][0]+data[0][1],10)) \
                                                                           + '</td></tr></table>')

        f.write('\n\t<br><p style=\'margin-left:20px;font-size:10px;\'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;most significant amplitudes<br>')
        f.write('\n\t<br><table>\n\t\t\t<tr><td><b>t<sup>a</sup><sub>i</sub></b></td></tr>')
        for i in data[1][0]:
            if not '0.0 ' in i: f.write('\n\t\t\t<tr><td>' + i + '</td></tr>')
        f.write('\n\t\t\t<tr><td><b>t<sup>ab</sup><sub>ij</sub></b></td></tr>')
        for i in data[1][1]:
            if not '0.0 ' in i: f.write('\n\t\t\t<tr><td>' + i + '</td></tr>')
        f.write('\n\t</table>')
        f.write('\n\t<table><caption>eom-ccsd</caption>')
        l = data[4][0]
        u = data[4][1]
        n = 0
        for i,e in enumerate(data[2]):
            if e[0] > l and e[0] < u:
                f.write('\n\t\t<td>' + "%.6f" % round(e[0], 6) + ' (' + e[1] + ')</td>')
                n += 1
            if ((n % 6) == 5) and (n != len(data[2])):
                f.write('\n\t\t</tr>\n\t\t<tr>')
        f.write('\n\t\t</tr>\n\t</table>')
        f.write('\n\t<br><table><caption>eom-mbpt(2)</caption>')
        n = 0
        for i,e in enumerate(data[3]):
            if e[0] > l and e[0] < u:
                f.write('\n\t\t<td>' + "%.6f" % round(e[0], 6) + ' (' + e[1] + ')</td>')
                n += 1
            if ((n % 6) == 5) and (n != len(data[3])):
                f.write('\n\t\t</tr>\n\t\t<tr>')
        f.write('\n\t\t</tr>\n\t</table>')

    if type == 'cogus':
        delimiter(f)
        f.write('\n\t<table><caption>Cluster Operator Generator Using Sympy</caption>')
        for cluster in data:
            if cluster[2] != 0.0:
                f.write('\n\t\t<tr><td>' + cluster[0] + '</td><td>' + str(round(cluster[1],10)) + '</td><td>' +  \
                     str(round(cluster[2],10)) + '</td></tr>')
            else:
                f.write('\n\t\t<tr><td>' + cluster[0] + '</td><td>' + str(round(cluster[1],10)) + '</td><td>' +  \
                        '' + '</td></tr>')

        f.write('\n\t</table>')
    f.close()

def post(exit = True):
    f = open('harpy.html','a')
    if not exit:
        f.write('\n\t<br><p style=\'margin-left:40px;font-size:10px;\'><b>SCF failed to converged in maximum cycles</b></p>')
    f.write('\n</body>')
    f.close()

def showMatrix(title, matrix, f, precision = '%.4f'):
    f.write('\n\t<br>\n\t<table>\n\t\t<caption>' + title + '</caption>\n\t\t<tr><th></th>')
    for i in range(0, matrix.shape[1]):
        f.write('\n\t\t<th>' + str(i+1) + '</th>',)
    f.write('</tr>')    
    for i in range(0, matrix.shape[0]):
        f.write('\n\t\t<tr><td>' + str(i+1) + '</td>',)
        for j in range(0, matrix.shape[1]):
            f.write('\n\t\t<td>' + precision % round(matrix[i,j],int(precision[2])) + '</td>',)
        f.write('</tr>')
    f.write('\n\t\t</tr>\n\t</table>')


def matrixHeatPlot(a, title=''):
    #display a heat map of matrix 'a'

    py.title(title)
    py.imshow(a, interpolation = 'nearest', cmap='jet', alpha=0.5)
    py.colorbar()
    py.show()

def evaluateGaussian(iBasis, x ,y , plane, z, normal):
    #compute electron distribution due to a Gaussian wavefunction

    planes = { 'xy' : (0,1,2), 'yz' : (1,2,0), 'zx' : (2,0,1)}
    axes = planes[plane]

    density = 0.0

    r = (x - iBasis.center[axes[0]])*(x - iBasis.center[axes[0]]) + \
        (y - iBasis.center[axes[1]])*(y - iBasis.center[axes[1]]) + \
        (z - iBasis.center[axes[2]])*(z - iBasis.center[axes[2]]) 

    c = (x - iBasis.center[axes[0]])**(iBasis.momentum[axes[0]]) * \
        (y - iBasis.center[axes[1]])**(iBasis.momentum[axes[1]]) * \
        (z - iBasis.center[axes[2]])**(iBasis.momentum[axes[2]]) 

    for i in range(0, len(iBasis.co)):
        prims = iBasis.co[i]
        exp = math.exp(-iBasis.ex[i]*r)
        
        if normal: density += prims * c * exp * iBasis.normal[i]
        else: density += prims * c * exp

    return density


def plotGaussianOverlap(iBasis, jBasis, plane, z, extent, grid, atoms, options = [False, 20]):
    #make matrix of plotting points and plot contours
    # plane = 'xy'|'yz'|'zx'    z displacement from plane, limits [min, max, min, max]

    planes = { 'xy' : (0,1,2), 'yz' : (1,2,0), 'zx' : (2,0,1)}
    planeAxes = planes[plane]

    normalise, nContours = options

    fig = py.figure(figsize=(5,5))
    ax = fig.gca()

    if jBasis == None: 
        ax.set_title('atom ' + atoms[iBasis.atom].id + ' orbitals ' + iBasis.symbol)
    else:
        ax.set_title(atoms[iBasis.atom].id + '(' + iBasis.symbol + ') ' + atoms[jBasis.atom].id + '(' + jBasis.symbol + ')')

    for i in range(0, len(atoms)):
        c = py.Circle(([atoms[i].center[planeAxes[0]], atoms[i].center[planeAxes[1]]]), \
            covalentRadius[atoms[i].number-1]/500, color = '0')
        ax.add_artist(c)
        for j in range(i+1, len(atoms)):
            if isBond(atoms,i,j):
                py.plot([atoms[i].center[planeAxes[0]],atoms[j].center[planeAxes[0]]], \
                        [atoms[i].center[planeAxes[1]],atoms[j].center[planeAxes[1]]], color = 'k')

    #grid is mesh size, generate grid points
    x = linspace(extent[0], extent[1], grid)
    y = linspace(extent[2], extent[3], grid)

    #get values
    Z = zeros((grid, grid))
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            Z[j,i] = evaluateGaussian(iBasis, x[i], y[j], plane, z, normalise) 
            if jBasis != None: Z[j,i] *= evaluateGaussian(jBasis, x[i], y[j], plane, z, normalise)

    heights = linspace(min([min(i) for i in Z]), max([max(i) for i in Z]), nContours)

    if iBasis == jBasis:
        map = 'Reds'
    else:
        map = 'seismic'

    try:
        cs = py.contour(x, y, Z, heights, cmap = map)
        py.text(extent[0]+0.4, extent[2]+0.4, 'plane is ' + str(plane) + ', elevation ' + str(z))       
        py.show()
    except:
        print('no density')

def plotMO(C, orbital, plane, z, extent, grid, atoms, bases, options = [True, 60, 1e-8]):
    #make matrix of plotting points and plot contours
    # plane = 'xy'|'yz'|'zx'    z displacement from plane, limits [min, max, min, max]

    planes = { 'xy' : (0,1,2), 'yz' : (1,2,0), 'zx' : (2,0,1)}
    planeAxes = planes[plane]

    #options
    normalise, nContours, cutoff  = options

    fig = py.figure(figsize=(5,5))
    ax = fig.gca()
    ax.set_title('molecular orbital ' + str(orbital))

    #draw molecule
    for i in range(0, len(atoms)):
        c = py.Circle(([atoms[i].center[planeAxes[0]], atoms[i].center[planeAxes[1]]]), \
            covalentRadius[atoms[i].number-1]/500, color = '0')
        ax.add_artist(c)
        for j in range(i+1, len(atoms)):
            if isBond(atoms,i,j):
                py.plot([atoms[i].center[planeAxes[0]],atoms[j].center[planeAxes[0]]], \
                         [atoms[i].center[planeAxes[1]],atoms[j].center[planeAxes[1]]], color = 'k')

    #grid is mesh size, generate grid points
    x = linspace(extent[0], extent[1], grid)
    y = linspace(extent[2], extent[3], grid)

    mo = C[:, orbital]

    #get values
    Z = zeros((grid, grid))
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            Z[j,i] = 0.0
            for m in range(0, len(mo)):
                amplitude = evaluateGaussian(bases[m], x[i], y[j], plane, z, normalise) * mo[m]
                if abs(amplitude) > cutoff: Z[j,i] += amplitude

    heights = linspace(min([min(i) for i in Z]), max([max(i) for i in Z]),nContours)

    try:
        cs = py.contourf(x, y, Z, heights, cmap = 'seismic')
        py.text(extent[0]+0.4, extent[2]+0.4, 'plane is ' + str(plane) + ', elevation ' + str(z))       
        py.show()
    except:
        print('no density')
        
