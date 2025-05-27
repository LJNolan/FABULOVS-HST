#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 09:54:53 2022

@author: Liam Nolan
"""

def isInt(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


def getElements(file): 
   with open('input/' + file, 'r') as inpt: # read in all lines of an input file
      lines = [line.strip() for line in inpt]
      
      lines = [line for line in lines if line] # clean empty lines
      
      # Make a list of lists of variables for each component of the input model
      ellist = []
      compiling = False
      elements = []
      for line in lines:
         if compiling:
            if line[0] == 'Z':
               compiling = False
               ellist.append(elements)
            else:
               elements.append(int(line.partition(")")[0]))
         if line[0] == '0':
            compiling = True
            elements = [line.split(' ')[1]]
            
      return(ellist)


def toConstraints(ellist):
   compcon = []
   for component in ellist:
      constraints = [component[0]]
      for element in component[1:]:
         if element == 1:
            if component[0] == 'sky':
               constraints.append(str(element))
            else:
               constraints.append('x')
               constraints.append('y')
         elif element > 1:
            constraints.append(str(element))
      compcon.append(constraints)
   return compcon


def makeConstraints(compcon):
   with open('constraint_ex', 'r') as inpt:
      lines = [line.rstrip() for line in inpt]
      opener = lines[0:2]
   
   for n in range(len(compcon)):
      component = compcon[n]
      opener.append('')
      opener.append('# Component ' + str(n + 1) + ': ' + component[0])
      for element in component[1:]:
         if isInt(element):
            if int(element) < 10:
               opener.append('#     ' + str(n + 1) + '             ' + element + '           1       # ')
            else:
               opener.append('#     ' + str(n + 1) + '             ' + element + '          1       # ')
         else:
            opener.append('#     ' + str(n + 1) + '             ' + element + '           1       # ')
   
   opener.append('')
   opener.append('# refrigerator')
   return opener


def doit(file):
   lines = makeConstraints(toConstraints(getElements(file)))
   with open('constraints/' + file, 'w') as feel:
      for line in lines:
         feel.write("%s\n" % line)
   return


b = 4700
while b < 4708:
   b += 1
   doit(str(b))
