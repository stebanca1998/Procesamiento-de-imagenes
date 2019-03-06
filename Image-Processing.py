import pydicom
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import ImageTk, Image
from math import fabs
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename

#--------------------------Variables globales-------------------------------
filename = ""
intensity = [0]*65536
rows = 0
columns = 0
matrizSobelX=[[-1,0,1],[-2,0,2],[-1,0,1]] #Matriz gradiente en X
matrizSobelY=[[-1,-2,-1],[0,0,0],[1,2,1]] #Matriz gradiente en Y

#----------------------------------Funciones---------------------------------

#--Selecciona el archivo y habilita botones de funciones
def selectFile():
	filename = filedialog.askopenfilename()
	global ds,rows,columns,data
	
	ds = pydicom.dcmread(filename)
	rows = int(ds.Rows)
	columns = int(ds.Columns)


	if filename!="":
		look.config(state="normal")
		histogram.config(state="normal")
		gaussian.config(state="normal")
		raleygh.config(state="normal")
		fGaussE.config(state="normal")
		mediana.config(state="normal")
		bordes.config(state="normal")

#--Abrir la imagen
def openImage():
	plt.imshow(ds.pixel_array) #,cmap=plt.cm.gray
	plt.show()

#--Hacer e imprimir el histograma
def histogram():

	global intensity
	intensity=[0]*65536
	for i in range (rows):
		for j in range (columns):
			intensity[ds.pixel_array[i,j]]=intensity[ds.pixel_array[i,j]]+1

	intensity = np.asarray(intensity)
	plt.plot(intensity)
	plt.show()

def filtroGaussiano():
	matriz = getGaussianFilter(4)[0]
	escalar = getGaussianFilter(4)[1]

	copia = ds.pixel_array.copy() 
	copia = convolucion(copia,matriz,escalar)

	plt.imshow(copia)
	plt.show()

def filtroRayleigh():
	matriz = getRayleighFilter(3)[0]
	escalar = getRayleighFilter(3)[1]

	copia = ds.pixel_array.copy() 
	copia = convolucion(copia,matriz,escalar)

	plt.imshow(copia)
	plt.show()

def filtroGaussianoEntero():
	matriz = getIntegerValuedGaussianFilter()[0]
	escalar = getIntegerValuedGaussianFilter()[1]

	copia = ds.pixel_array.copy() 
	copia = convolucion(copia,matriz,escalar)

	print(ds[100][100])
	print(copia[100][100])

	plt.imshow(copia)
	plt.show()

def hallarBordes():
	copia = ds.pixel_array.copy()

	gradienteX = convolucion(copia,matrizSobelX,1)
	gradienteY = convolucion(copia,matrizSobelY,1)

	copia = seleccionarBordes(copia,gradienteX,gradienteY)

	plt.imshow(copia)
	plt.show()

def aplicarMediana():
	copia = ds.pixel_array.copy()

	copia = filtroMediana(copia)

	plt.imshow(copia)
	plt.show()


#------------------------------Funciones adicionales----------------------------------

#--Obtener kernel gaussiano
def getGaussianFilter(neighbours=1, sigma=1):
	N = neighbours*2+1
	X = Y = np.zeros((N, N))

	Y[:] = np.arange(N,dtype=np.float32)-neighbours
	X = np.transpose(Y) 
	
	gaussianFilter = np.zeros((N, N))

	leftSide = (1 / (np.pi * 2 * np.power(sigma,2)))

	aux = -(np.power(X, 2) + np.power(Y, 2)) / (2 * np.power(sigma, 2))
	rightSide = np.exp(aux)

	gaussianFilter = leftSide*rightSide
	factorValue = 1/np.sum(gaussianFilter)

	return gaussianFilter,factorValue

#--Obtener kernel rayleigh
def getRayleighFilter(neighbours=1, sigma=1):
	N =  neighbours * 2 + 1
	X = Y = np.zeros((N, N))

	Y[:] = np.arange(N, dtype=np.float32)
	X = np.transpose(Y)

	rayleighFilter = np.zeros((N, N))
	leftSide = np.divide(X*Y, np.power(sigma,4))

	aux = -(np.power(X, 2) + np.power(Y, 2)) / (2 * np.power(sigma, 2))
	rightSide = np.exp(aux)

	rayleighFilter = leftSide*rightSide
	factorValue = 1/np.sum(rayleighFilter)

	return rayleighFilter,factorValue

#--Obtener kernel gaussiano entero
def getIntegerValuedGaussianFilter(neighbours = 1, sigma = 1):
	N = neighbours*2 + 1

	pascalRow = np.asarray(getKthPascalRow(N - 1), dtype=np.int32)
	pascalRow = np.expand_dims(pascalRow,1)

	extendedPascal = np.resize(pascalRow,(pascalRow.shape[0],pascalRow.shape[0]))

	result = pascalRow * extendedPascal
	scalarFactor = np.sum(result)

	return result,scalarFactor

#--Función para hallar el kernel gaussiano de enteros
def getKthPascalRow(rowNumber):
	if rowNumber == 0:
		return [1, ]

	lastRow = [1, ]
	for R in range(1, rowNumber+1):
		row = []
		row.append(1)
		for C in range(R - 1):
			row.append(lastRow[C] + lastRow[C+1])
		row.append(1)
        
		lastRow = row
	return lastRow

def convolucion(imagen,matriz,escalar):
	imagenM = imagen
	s = len(matriz)#Tamaño del Kernel
	n = (s-1)/2#numero de vecinos

	for i in range(0,rows):
		for j in range(0,columns):
			if i<n or i>(rows-n) or j<n or j>(columns-n):
				imagenM[i][j]=0
			else:
				p = 0
				q = 0
				aux = 0
				for x in range(int(i-n),int(i+n)):
					for y in range(int(j-n),int(j+n)):
						aux+=imagen[x][y]*matriz[p][q]
						q+=1#end for y
					p+=1
					q=0#end for x

				aux = aux/escalar
				aux = int(aux)
				imagenM[i][j] = aux
			#end if
		#end for j
	#end for i
	return imagenM


def seleccionarBordes(imagen,matrizX,matrizY):
	for i in range(rows):
		for j in range(columns):
			if (fabs(matrizX[i,j])+fabs(matrizY[i,j])>10000):
				imagen[i,j]=0
			else:
				imagen[i,j]=1
	return imagen

def filtroMediana(imagen, n = 1):
	for i in range(rows-1):
		for j in range(columns-1):
			if i<n or i>((rows-1)-n) or j<n or j>((columns-1)-n):
				imagen[i][j]=0
			else:
				list=[]
				for x in range(int(i-n),int(i+n)):
					for y in range(int(j-n),int(j+n)):
						list.append(imagen[x,y])

	return imagen

def ordenar(lista):
	n = len(lista)

	for i in range (n):
		for j in range (0, n-i-1):
			if lista[j] > lista[j+1]:
				aux = lista[j]
				lista[j] = lista[j+1]
				lista[j+1] = aux
	return lista

def otsu():
	umbral=0
	total = rows*columns
	sum = 0
	for i in range (0,65536,1):
		sum = sum + i*intensity[i]

	sumB = 0
	wb = 0
	wf = 0

	varmax = 0
	umbral = 0

	for i in range (50,14000,1):
		wb = wb + intensity[i]
		if (wb==0):
			continue

		wf = total - wb
		if(wf==0):
			break

		sumB = sumB + (i*intensity[i])
		mb = sumB/wb
		mf = (sum - sumB)/wf

		varBetween = float(wb)*float(wf)*(mb-mf)*(mb-mf)

		if (varBetween > varmax):
			varmax = varBetween
			umbral = i
	#end for
	return umbral


#------------------------------------GUI-----------------------------------

ventanap = Tk()
ventanap.title("DICOM Image Processing")
ventanap.geometry("450x250")

myFrame = Frame(ventanap)

seleccionador = Button(myFrame,text="Seleccionar imagen", command=selectFile)
seleccionador.grid(row=1,column=1)

look = Button(myFrame, text = "Mostrar imagen", command=openImage, state = DISABLED)
look.grid(row=2,column=1)

histogram = Button(myFrame,text="Hacer histograma", command=histogram, state = DISABLED)
histogram.grid(row=3,column=1)

gaussian = Button(myFrame,text="Filtro Gaussiano", command=filtroGaussiano, state = DISABLED)
gaussian.grid(row=4,column=1)

raleygh = Button(myFrame,text="Filtro Raleygh", command=filtroRayleigh, state = DISABLED)
raleygh.grid(row=5,column=1)

fGaussE = Button(myFrame,text="Filtro Gaussiano entero", command=filtroGaussianoEntero, state = DISABLED)
fGaussE.grid(row=6,column=1)

mediana = Button(myFrame,text="Filtro mediana", command=aplicarMediana, state = DISABLED)
mediana.grid(row=7,column=1)

bordes = Button(myFrame,text="Bordes(Sobel)", command=hallarBordes, state = DISABLED)
bordes.grid(row=8,column=1)

myFrame.pack()

ventanap.mainloop()