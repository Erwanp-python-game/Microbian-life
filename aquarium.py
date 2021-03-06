import pygame
from math import *
import numpy as np
from pygame.locals import *
from random import *
import matplotlib.pyplot as plt
from copy import*
from scipy import signal
for i in range(1,10):
	a=0
	for j in range(0,i):
		a=a+abs(cos(2*pi*(j+0.5)/i))
	b=0
	for j in range(0,i):
		b=b+abs(cos(2*pi*(j)/i))
	print(max(a,b),i)

def randomRot(x,y):
	a=(random()-0.5)*pi/2
	x2=x*cos(a)+y*sin(a)
	y2=-x*sin(a)+y*cos(a)
	return x2,y2
L=600
pygame.init()
fenetre = pygame.display.set_mode((L+200, L))
print(pygame.time.get_ticks())



courantX=np.full((L,L),0.0)
courantY=np.full((L,L),0.0)

courantX0=np.full((3*L,3*L),0.0)
courantY0=np.full((3*L,3*L),0.0)

x,y = np.meshgrid(np.linspace(-30,30,3*L),np.linspace(-30,30,3*L))
speed=2.5# previous3
Rs=randint(2,3)#5
print(Rs)
for i in range(0,Rs):
	xR=randint(-10,10)
	yR=randint(-10,10)
	I=(-1)**randint(0,1)
	Rayon=randint(4,15)
	
	z2=speed*(I*(x-xR)/(((x-xR)**2+(y-yR)**2)**0.5))*np.exp(-((((x-xR)**2+(y-yR)**2)-Rayon**2)**2)/(60**2))
	z1=speed*(-I*(y-yR)/(((x-xR)**2+(y-yR)**2)**0.5))*np.exp(-((((x-xR)**2+(y-yR)**2)-Rayon**2)**2)/(60**2))
	courantX0=np.add(courantX0,np.array(z1))
	courantY0=np.add(courantY0,np.array(z2))

for i in range(0,3):
	for j in range(0,3):
		courantX=np.add(courantX0[i*L:(i+1)*L,j*L:(j+1)*L],courantX)
		courantY=np.add(courantY0[i*L:(i+1)*L,j*L:(j+1)*L],courantY)
		
# for i in range(0,L):
	# for j in range(0,L):
		# courantX[i][j],courantY[i][j]=randomRot(courantX[i][j],courantY[i][j])

print(pygame.time.get_ticks())
Lco=10
filterB=np.full((Lco,Lco),0.0)
for i in range(0,Lco):
	for j in range(0,Lco):
		filterB[i][j]=np.exp((-0.5*(((i-Lco/2)**2+(j-Lco/2)**2)**2)/(Lco**2)))*(1/((2*pi)*Lco))
filterB=filterB/filterB.sum()



courantX=signal.convolve2d(courantX, filterB, mode='same', boundary='wrap')
courantY=signal.convolve2d(courantY, filterB, mode='same', boundary='wrap')
# courantX=courantX0[L:2*L,L:2*L]
# courantY=courantY0[L:2*L,L:2*L]

subL=60
proie=np.full((subL,subL),1.0)
proie_mem=np.full((subL,subL),1.0)

proie_dict={}
pred=np.full((subL,subL),1.0)
pred_mem=np.full((subL,subL),1.0)

dt=0.1
diffu=0.1
nourriture=np.random.uniform(size=subL**2)*4
flux=np.full((subL**2,subL**2),0.0)

for i in range(0,subL**2):
		flux[i][i]=(1/dt)-abs(courantY[i%subL][i//subL])-abs(courantX[i%subL][i//subL])-4*diffu/dt
		
		
		flux[i][(i+1)%(subL**2)]=0.5*abs(courantX[10*((i)//subL)][10*(((i+1)%(subL**2))%subL)]-abs(courantX[10*((i)//subL)][10*(((i+1)%(subL**2))%subL)]))+diffu/dt
		flux[i][(i-1)%(subL**2)]=0.5*abs(courantX[10*((i)//subL)][10*(((i-1)%(subL**2))%subL)]+abs(courantX[10*((i)//subL)][10*(((i-1)%(subL**2))%subL)]))+diffu/dt

		
		flux[i][(i+subL)%(subL**2)]=0.5*abs(courantY[10*(((i+subL)%(subL**2))//subL)][10*(i%subL)]-abs(courantY[10*(((i+subL)%(subL**2))//subL)][10*((i)%subL)]))+diffu/dt
		flux[i][(i-subL)%(subL**2)]=0.5*abs(courantY[10*(((i-subL)%(subL**2))//subL)][10*(i%subL)]+abs(courantY[10*(((i-subL)%(subL**2))//subL)][10*((i)%subL)]))+diffu/dt



flux=flux*dt

q=0
trace=0
clock = pygame.time.Clock()
CO2=30000
O2=30000
C_dissou=30000

images_cell={}
images_cell['M']=pygame.image.load('M.png')
images_cell['P']=pygame.image.load('P.png')
images_cell['G']=pygame.image.load('G.png')
images_cell['N']=pygame.image.load('N.png')
images_cell['Y']=pygame.image.load('Y.png')
images_cell['R']=pygame.image.load('R.png')
images_cell['O']=pygame.image.load('O.png')
images_cell['C']=pygame.image.load('C.png')
images_cell['S']=pygame.image.load('S.png')
images_cell['E']=pygame.image.load('E.png')
images_cell_center=['M','P','G','R','O','C','E']
images_cell_end=['N','Y','S']#,'Y','P','C']

all_codes=[]
seed(0)
nbc={}
nbc['M']=randint(0,200)
nbc['P']=randint(0,200)
nbc['G']=randint(0,200)
nbc['N']=randint(0,200)
nbc['Y']=randint(0,200)
nbc['R']=randint(0,200)
nbc['O']=randint(0,200)
nbc['C']=randint(0,200)
nbc['S']=randint(0,200)
nbc['E']=randint(0,200)

fleche=pygame.image.load('fleche.png')
fond=pygame.Surface((L,L),pygame.SRCALPHA, 32)
fond.fill((0,50,200))
for i in range(0,30):
	for j in range(0,30):
		if courantX[i*20][j*20]>0:
			angle=np.arctan(courantY[i*20][j*20]/courantX[i*20][j*20])*360/(2*pi)+180
		else:
			angle=np.arctan(courantY[i*20][j*20]/courantX[i*20][j*20])*360/(2*pi)
		flecheR=pygame.transform.rotate(fleche,angle)
		S=int((10/speed)*(courantX[i*20][j*20]**2+courantY[i*20][j*20]**2)**0.5)
		flecheR=pygame.transform.scale(flecheR,(S,S))
		fond.blit(flecheR,(20*i,20*j))
fond0=fond.copy()
		
def AngleReturn(x,y):
	if x<0:
		return np.arctan(y/x)*(180/pi)+180
	else:
		return np.arctan(y/x)*(180/pi)

def meanAngle(a1,a2,size):
	a1=a1/(180/pi)
	a2=a2/(180/pi)
	S=min(size/1000,0.05)
	return AngleReturn((0.9+S)*cos(a1)+(0.1-S)*cos(a2),(0.9+S)*sin(a1)+(0.1-S)*sin(a2))

def show_food(actif):
	if actif==1:
		if trace==1:
			fond2=fond.copy()
		else:
			fond2=fond0.copy()
		listim=[]
		im=pygame.Surface((L//subL,L//subL),pygame.SRCALPHA, 32)
		for i in range(0,subL**2):
			im.fill((0,min(max(int(200*nourriture[i]),0),255),200,100))
			
			listim.append((im.copy(),((L//subL)*(i//subL),(L//subL)*(i%subL))))
		
		fond2.blits(listim)
		return fond2
	else:
		return fond

def grad(I,yeux):
	X=np.full((2*yeux+1,2*yeux+1),0.0)
	Pr=np.full((2*yeux+1,2*yeux+1),1.0)
	for i in range(-yeux,yeux+1):
		for j in range(-yeux,yeux+1):
			X[i+yeux][j+yeux]=max(nourriture[(I+i+60*j)%3600]-nourriture[(I)%3600],0)#
			Pr[i+yeux][j+yeux]+=proie_mem[(I//subL+i)%subL][(I%subL+j)%subL]*((i-yeux)**2+(j-yeux)**2)**0.5
	X=np.divide(X,Pr.T)

	pos=np.unravel_index(np.argmax(X, axis=None), X.shape)
	Gl=2*max(3-yeux,1)
	return (pos[0]-yeux+randint(-Gl,Gl),pos[1]-yeux+randint(-Gl,Gl))
	
def grad_pred(I,yeux,vx,vy,D):
	X=np.full((2*yeux+1,2*yeux+1),0.0)
	for i in range(-yeux,yeux+1):
		for j in range(-yeux,yeux+1):
			X[i+yeux][j+yeux]=max(proie_mem[(I//subL+i)%subL][(I%subL+j)%subL]-proie_mem[(I//subL)%subL][(I%subL)%subL],0)#
	pos=np.unravel_index(np.argmax(X.T, axis=None), X.shape)
	Gl=2*max(3-yeux,1)
	Ra=1.5*pi*random()
	if np.sum(X)!=0 and D==0:
		return (pos[0]-yeux+randint(-Gl,Gl),pos[1]-yeux+randint(-Gl,Gl))
	else:
		return (pos[0]*0,pos[1]*0)#(pos[0]*0+vx*cos(Ra)+vy*sin(Ra),pos[1]*0+vy*cos(Ra)-vx*sin(Ra))


def grad_proie(I,yeux):
	X=np.full((2*yeux+1,2*yeux+1),0.0)
	for i in range(-yeux,yeux+1):
		for j in range(-yeux,yeux+1):
			X[i+yeux][j+yeux]=max(pred_mem[(I//subL+i)%subL][(I%subL+j)%subL]-pred_mem[(I//subL)%subL][(I%subL)%subL],0)#
			
	pos=np.unravel_index(np.argmax(X.T, axis=None), X.shape)
	Gl=2*max(3-yeux,1)
	if np.sum(X)!=0:
		return (pos[0]-yeux+randint(-Gl,Gl),pos[1]-yeux+randint(-Gl,Gl))
	else:
		return (pos[0]*0,pos[1]*0)
	
All_Org=[]
class Organism():
	
	def __init__(self,x,y,code,Img):
		self.xc=x
		self.yc=y
		self.vx=0
		self.vy=0
		self.age=0
		self.code=code
		self.imL=Img
		self.im=Img[-1]
		self.type_nourr=0
		self.type_photo=0
		self.gras=0
		self.angle=0
		self.yeux=0
		self.nageoire=0
		seedn=0
		self.fast=0
		self.racine=0
		self.born=I
		self.os=0
		self.bouche=0
		self.pic=0
		self.esto=0
		addL=0
		self.sym=self.code[0][0]
		for i in code:
			for j in i:
				if 'M'==j:
					self.type_nourr+=1
					self.fast=0.1
				if 'P'==j:
					self.type_photo+=1
				if 'G'==j:
					self.gras+=0.5
				if 'Y'==j:
					self.yeux+=1
				if 'N'==j:
					self.nageoire+=1
				if 'O'==j:
					self.os+=1
				if 'C'==j:
					self.bouche+=1
				if 'R'==j:
					self.racine+=1
					self.type_photo+=1
				if 'S'==j:
					self.pic+=1
				if 'E'==j:
					self.esto+=1
			addL=addL+2*(len(i)-1)+1
		addL=addL-self.nageoire*1.5-self.yeux*1.5-self.pic*1.5
		for j in range(1,len(code)):
			seedn+=nbc[self.code[j][0]]
		seed(seedn)
		self.color=(randint(0,255),randint(0,255),randint(0,255))
		seed()
		
		self.size=(addL-2)*self.sym+1
		self.type_nourr=max((self.type_nourr-1)*self.sym+1,0)# 
		self.type_photo=max((self.type_photo-1)*self.sym+1,0)# gras pas assez efficace
		self.gras=max((self.gras-0.5)*self.sym+0.5,0)
		self.os=max((self.os-1)*self.sym+1,0)
		self.esto=max((self.esto-1)*self.sym+1,0)
		self.pic=max((self.pic-1)*self.sym+1,0)
		self.bouche=max((self.bouche-1)*self.sym+1,0)
		self.yeux=int(max((self.yeux-1)*self.sym+1,0)*np.heaviside(self.fast,0))
		self.nageoire=int(max((self.nageoire-1)*self.sym+1,0)*np.heaviside(self.fast,0))
		self.stockedCO2=250+50*self.size#300 de base
		self.not_fixed=1
		self.digestion=0
		
	def buil_Im(self):
		Lm=max(30,20*(len(self.code)),20*max(len(elem) for elem in self.code))
		self.imL=[]
		for h in range(7,int(np.heaviside(self.nageoire+self.bouche,0))*30+1+7):
			ImA=pygame.Surface((Lm,Lm),pygame.SRCALPHA, 32)# take width into account
			
			for j in range(0,self.sym):
				for i in range(1,len(self.code)):
					A=j*2*pi/self.sym
					X=int((i-1)*10*sin(A))+Lm//2-5
					Y=int((i-1)*10*cos(A))+Lm//2-5
					if self.code[i][0] in ['E','C']:
						IMR=pygame.transform.rotate(images_cell[self.code[i][0]],A*180/pi)
						X+=5-IMR.get_width()//2
						Y+=5-IMR.get_height()//2
					else:
						IMR=images_cell[self.code[i][0]]
					
					
					ImA.blit(IMR,(X,Y))# un peu shifted ?? prendre en compte
					for k in range(1,len(self.code[i])):
						deltaX=int((k)*10*cos(A))
						deltaY=-int((k)*10*sin(A))
						if self.code[i][k]=='N':
							Anage=10*abs(h%30-15)-75
							supX1=(5*cos((pi/180)*(-Anage))-5)*np.heaviside(self.nageoire+self.bouche,0)#attention on fait tourner les yeux aussi
							supY1=5*sin((pi/180)*(-Anage))*np.heaviside(self.nageoire+self.bouche,0)
							supX=int(supX1*cos(A)+supY1*sin(A))
							supY=int(-supX1*sin(A)+supY1*cos(A))
							supX2=int(-supX1*cos(A)+supY1*sin(A))
							supY2=int(+supX1*sin(A)+supY1*cos(A))
						else:
							Anage=0
							supX=0
							supY=0
							supX2=0
							supY2=0
						if self.code[i][k] not in images_cell_end:
							if self.code[i][0] in ['E','C']:
								IMR=pygame.transform.rotate(images_cell[self.code[i][k]],A*180/pi)
							else:
								IMR=images_cell[self.code[i][k]]
							ImA.blit(IMR,(X+deltaX,Y+deltaY))
							ImA.blit(IMR,(X-deltaX,Y-deltaY))
						else:
							Im=pygame.transform.rotate(images_cell[self.code[i][k]],A*180/pi+Anage)
							shift=(Im.get_width()//2)-5
							Ir=pygame.transform.flip(images_cell[self.code[i][k]],True,True)
							ImA.blit(Im,(X+deltaX-shift+supX,Y+deltaY-shift+supY))
							ImA.blit(pygame.transform.rotate(Ir,A*180/pi-Anage),(X-deltaX-shift+supX2,Y-deltaY-shift+supY2))
							
			ImA.fill((255,255,255,215),special_flags=BLEND_RGBA_MULT)
			self.im=ImA.copy()
			self.imL.append(ImA)
		if len(all_codes)==len(all_im)+1 and self.code==all_codes[-1]:
			all_im.append(self.im)
		
	def move_eat(self):
		global CO2,O2,nourriture,All_Org,proie
		I1=int(self.yc)//10+60*(int(self.xc)//10)
		G=grad(I1,self.yeux+1)
		G=G/((G[0]**2+G[1]**2)**0.5+0.0001)
		G2=(0,0)
		G3=(0,0)
		if self.bouche>0:
			G2=grad_pred(I1,self.yeux+1,self.vx,self.vy,self.digestion)
			G2=G2/((G2[0]**2+G2[1]**2)**0.5+0.0001)
		else:
			G3=grad_proie(I1,self.yeux+1)
			G3=G3/((G3[0]**2+G3[1]**2)**0.5+0.0001)
		
		self.vx=(0.5*self.vx+0.3*courantY[int(self.xc)][int(self.yc)]/(log(self.size/10+3)*(1*self.nageoire+2)))+(10+2*self.nageoire)*dt*((G[1]-1.1*G3[1]*(1-np.heaviside(self.bouche,0)))*np.heaviside(self.fast,0)+G2[1]*np.heaviside(self.bouche,0))/log(self.size/20+3)+(1-np.heaviside(self.fast,0))*np.random.normal(0,0.2)# div norme de G
		self.vy=(0.5*self.vy+0.3*courantX[int(self.xc)][int(self.yc)]/(log(self.size/10+3)*(1*self.nageoire+2)))+(10+2*self.nageoire)*dt*((G[0]-1.1*G3[0]*(1-np.heaviside(self.bouche,0)))*np.heaviside(self.fast,0)+G2[0]*np.heaviside(self.bouche,0))/log(self.size/20+3)+(1-np.heaviside(self.fast,0))*np.random.normal(0,0.2)# viv v par taille
		self.xc=(self.xc+self.vx*self.not_fixed)%L
		self.yc=(self.yc+self.vy*self.not_fixed)%L
		consumed=0
		if self.type_nourr>0:
			consumed=max(min(nourriture[I1],10*self.type_nourr,10*self.type_nourr*O2/15000),0)
			nourriture[I1]=nourriture[I1]-consumed
			CO2=CO2+consumed/2
			O2=O2-consumed
		if self.type_photo>0:
			consumed=max(min(10*self.type_photo,10*self.type_photo*CO2/15000),0)
			nourriture[I1]=nourriture[I1]+consumed/2
			CO2=CO2-consumed
			O2=O2+consumed/2
		
		self.stockedCO2=self.stockedCO2+consumed/2
		self.age=self.age+1+self.bouche*0.5+0.5*(9+0.05*self.size-consumed*(self.sym+0.2)/(self.sym))/(1+self.gras)
		self.I1=I1
		self.angle=meanAngle(self.angle,AngleReturn(self.vx,self.vy),self.size)
		
		if self.bouche==0:
			if self.type_nourr>0:
				proie[int(self.xc)//10][int(self.yc)//10]+=self.size
		else:
			pred[int(self.xc)//10][int(self.yc)//10]+=self.size
			if str(int(self.xc)//10)+str(int(self.yc)//10) in proie_dict.keys():
				cible=proie_dict[str(int(self.xc)//10)+str(int(self.yc)//10)]
				if cible in All_Org and (self.bouche+(self.size//2)+randint(-2,2)>(cible.size//2)+(cible.os+cible.pic)*4 or randint(0,50)==0) and randint(0,30)==0:#randint(0,abs(int(10*cible.size*(1+1*cible.os))))<=self.size+self.bouche+1:
					Miam=cible.eated(self.bouche)# ??a un peu une bonne id??e !!!!!!!!!!!!!!!!!!!!!! avant al??atoire 300, 150 bien aussi
					self.digestion=Miam
					self.age-=0.5*(Miam*(self.sym+0.2)/(self.sym))/(1+self.gras)
			self.stockedCO2+=self.digestion-max(0,self.digestion-0.5-0.5*self.bouche-1*self.esto)
			self.digestion=max(0,self.digestion-0.5-0.5*self.bouche-1*self.esto)
			
		
		if self.stockedCO2>500+100*self.size and randint(0,10)==0:
			if randint(0,7)==0:
				CM=mutation(deepcopy(self.code))
				All_Org.append(Organism((self.xc+np.random.normal(0,5))%600,(self.yc+np.random.normal(0,5))%600,CM,self.imL))
				All_Org[-1].buil_Im()
				nb[all_codes.index(CM)]+=1
			else:
				All_Org.append(Organism((self.xc+np.random.normal(0,5))%600,(self.yc+np.random.normal(0,5))%600,self.code,self.imL))
				nb[all_codes.index(self.code)]+=1
			self.stockedCO2=self.stockedCO2-250-50*self.size
			
		if self.racine>0 and self.born+randint(100,300)<I and (self.vx**2+self.vy**2)<5+(self.racine*self.sym) and randint(0,5)==0:
			self.not_fixed=0	
				
	
	def draw(self,trace):
		global fenetre,all_codes,I
		if trace==1:
			pygame.draw.circle(fond,self.color,(int(self.xc-1),int(self.yc-1)),2)
		#pygame.draw.circle(fenetre,(self.type_nourr*255,(1-self.type_nourr)*255,0),(int(self.xc),int(self.yc)),4)
		Image=pygame.transform.rotate(self.imL[I%len(self.imL)],-self.angle-90)
		
		fenetre.blit(Image,(int(self.xc-Image.get_width()//2),int(self.yc-Image.get_height()//2)))
			
	
	def alive(self):
		return (self.age<2800+np.random.normal(0,100)+200*self.size+self.os*150)
	
	def eated(self,B):
		global CO2,O2,nourriture
		self.age=10**5
		mB=min(4-B,0)
		if randint(0,1)==0:
			nourriture[self.I1]=nourriture[self.I1]+mB*self.stockedCO2//4
		else:
			nourriture[self.I1]=nourriture[self.I1]+mB*self.stockedCO2//4
			O2=O2+mB*self.stockedCO2//4
		A=copy(self.stockedCO2-mB*self.stockedCO2//4)
		self.stockedCO2=0
		
		return A
		
	def release(self):
		global CO2,O2,nourriture
		if randint(0,1)==1:
			nourriture[self.I1]=nourriture[self.I1]+self.stockedCO2
			O2=O2+self.stockedCO2
		else:
			CO2=CO2+self.stockedCO2
		nb[all_codes.index(self.code)]-=1
		
		

all_codes.append([[1],['M']])
all_codes.append([[1],['P']])
all_im=[images_cell['M'],images_cell['P']]
col=[]

def update_colors():
	global col
	col=[]
	for code1 in all_codes:
		seedn=0
		for j in range(1,len(code1)):
			seedn+=nbc[code1[j][0]]
		seed(seedn)
		col.append((randint(0,255)/255,randint(0,255)/255,randint(0,255)/255))
update_colors()
nb=[0,0]
Big_data=[[0],[0]]
start_I=[0,0]

def showcode(code):
	a=''
	for i in code:
		a=a+'/'
		for j in i:
			a=a+str(j)
	return a

def mutation(code_g):# assurer qu'une mutation ait lieu
	muted=0
	code_r=deepcopy(code_g)
	while muted==0:
	
		R=randint(-10,11)
		if R>10:
			Muta=randint(1,len(code_g)-1)
			if code_r[Muta][-1] in images_cell_end:
				code_r[Muta][-1]=choice(images_cell_end)
				muted=1
		
		if R>8 and R<11:
			Muta=randint(1,len(code_g)-1)
			code_r[Muta][0]=choice(images_cell_center)
			muted=1
			
		if R<=8 and R>6:
			if randint(0,1)>=1:
				a=1
				for i in range(2,len(code_r)):
					a=min((i-1)*abs(tan(pi/(code_r[0][0]+1)))-len(code_r[i])+0.5,a)
				print('sym',a,code_r[0][0],code_g)
				if len(code_r)>2 and (code_r[0][0]==1 or a>=0) and len(code_r[1])==1:
					code_r[0][0]+=1
					muted=1
					print('sym_app')
				
			else:
				code_r[0][0]=max(code_r[0][0]-1,1)
				muted=1
			
		if R<=6 and R>-6:
	
			if len(code_r)==2 and len(code_r[1])==1:
				code_r[0][0]=choices([1,2,3,4,5,6],[6,5,4,3,2,1],k=1)[0] # faire r??tr??cir
			code_r.append([choice(images_cell_center)])
			muted=1
		
		if R<=-6 and R>-11:# check angle
				print("largeur")
				Muta=randint(1,len(code_g)-1)
				if (Muta-1)*abs(tan(pi/(code_g[0][0])))-1>len(code_g[Muta]) or code_g[0][0]<3:
					if randint(0,4)==1 or (code_r[Muta][-1] in images_cell_end):
						code_r[Muta].insert(0,choice(images_cell_center))
					else:
						code_r[Muta].append(choice(images_cell_end))
					muted=1
		
	if (code_r in all_codes)==False:
		all_codes.append(code_r)
		update_colors()
		nb.append(0)
		Big_data.append([0])
		start_I.append(I)
		print(showcode(code_r))
		return code_r
			
	else:
		return code_r

font = pygame.font.Font('freesansbold.ttf', 13)
def show_species(keyword):
	back=pygame.Surface((200,L), 32)
	back.fill((0,0,0))
	U=20-curseur*50
	fenetre.blit(back,(L,0))
	if keyword=='time':
		for i,j in enumerate(reversed(all_codes)):
			U=U+60
			T=showcode(j)
			text = font.render(T, True, (255,255,255))
			textRect = text.get_rect()
			textRect.topleft = (L+20, U)
			fenetre.blit(text,textRect)
			text = font.render(str(nb[len(all_im)-i-1]), True, (255,255,255))
			textRect = text.get_rect()
			textRect.topleft = (L+50, U+20)
			fenetre.blit(text,textRect)
			
			IM=pygame.transform.rotate(all_im[len(all_im)-i-1],90)
			IM=IM.convert(back)
			Ll=int(50*IM.get_width()/IM.get_height())
	
			pygame.draw.line(fenetre,(255,255,255),(L,U),(L+200,U))
	
			C=(int(col[len(all_im)-i-1][0]*255),int(col[len(all_im)-i-1][1]*255),int(col[len(all_im)-i-1][2]*255))
			pygame.draw.line(fenetre,C,(L+10,U+30),(L+20,U+30),5)
			fenetre.blit(pygame.transform.scale(IM,(Ll,50)),(textRect[0]+50,U+10))# afficher par plus vivantes et plus r??centes ??ventuellement plus complexes et plus vivantes au max
	
	if keyword=='nb':
		LE=[int(i) for _,i in reversed(sorted(zip(nb,np.linspace(0,len(nb)-1,len(nb)))))]
		
		for i in LE:
			j=all_codes[i]
			U=U+60
			T=showcode(j)
			text = font.render(T, True, (255,255,255))
			textRect = text.get_rect()
			textRect.topleft = (L+20, U)
			fenetre.blit(text,textRect)
			text = font.render(str(nb[i]), True, (255,255,255))
			textRect = text.get_rect()
			textRect.topleft = (L+50, U+20)
			fenetre.blit(text,textRect)
			
			IM=pygame.transform.rotate(all_im[i],90)
			IM=IM.convert(back)
			Ll=int(50*IM.get_width()/IM.get_height())
	
			pygame.draw.line(fenetre,(255,255,255),(L,U),(L+200,U))
	
			C=(int(col[i][0]*255),int(col[i][1]*255),int(col[i][2]*255))
			pygame.draw.line(fenetre,C,(L+10,U+30),(L+20,U+30),5)
			fenetre.blit(pygame.transform.scale(IM,(Ll,50)),(textRect[0]+50,U+10))# afficher par plus vivantes et plus r??centes ??ventuellement plus complexes et plus vivantes au max


for i in range(0,30):
	R=randint(0,1)
	if R==1:
		code=[[1],['M']]
	if R==0:
		code=[[1],['P']]


	All_Org.append(Organism(randint(0,L-1),randint(0,L-1),code,[0]))
	All_Org[-1].buil_Im()
	nb[all_codes.index(code)]+=1
	#Big_data[all_codes.index(code)][0]+=1/10
	
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)	
ax2 = fig.add_subplot(2, 1, 2)
graphe=0
graphe_tot=0
I=-1
fenetre.blit(fond,(0,0))
curseur=0
filterA=[[0,0.05,0],[0.05,0.8,0.05],[0,0.05,0]]

while q==0:
	T1=pygame.time.get_ticks()
	I+=1
	proie=np.full((subL,subL),1.0)
	pred=np.full((subL,subL),1.0)
	#pygame.time.wait(100)

	nourriture=np.dot(flux,nourriture)
	nourriture=np.absolute(nourriture)*(O2/np.absolute(nourriture).sum())
	
	for event in pygame.event.get():
		if event.type == QUIT:
			q=1
		KEY=pygame.key.get_pressed()
		if KEY[K_SPACE]:
			trace=(trace+1)%2
			print(trace)
		if KEY[K_a]:
			graphe=(graphe+1)%2
		if KEY[K_b]:
			graphe_tot=1
		if event.type==MOUSEBUTTONUP:
			if event.button==5:
				curseur=(curseur+1)
			if event.button==4:
				curseur=(curseur-1)

			

	Im=show_food(1)
	fenetre.blit(Im,(0,0))
	proie_dict_mem={}
	for i in All_Org:
		i.move_eat()
		if i.bouche==0 and i.type_nourr>0:
			proie_dict_mem[str(int(i.xc)//10)+str(int(i.yc)//10)]=i# bon d??but
		i.draw(trace)
		if i.alive()==False:
			i.release()
			All_Org.remove(i)
	proie_dict=proie_dict_mem.copy()
	proie=signal.convolve2d(proie, filterB, mode='same', boundary='wrap')
	proie_mem=proie.copy()
	pred_mem=pred.copy()
	
	if (I%100)==0 and graphe==1:
		ax2.scatter(I,O2,c='blue')
		ax2.scatter(I,CO2,c='red')
		for i in range(0,len(col)):
			ax1.scatter(I,nb[i],color=col[i])
	
	if graphe_tot==1:
		for i in range(0,len(col)):
			plt.plot(np.linspace(start_I[i],I,len(Big_data[i])),Big_data[i],color=col[i])
		plt.show()
		graphe_tot=0

	if I%10000==0:
		for i in range(0,len(col)):
			Big_data[i]=Big_data[i][::2]
		
		plt.pause(0.001)
	if (I%100)==1:
		print(pygame.time.get_ticks()-T1,O2+2*CO2+nourriture.sum()-300*2*10,np.sum(nb),np.sum(proie))
		for i in range(0,len(col)):
			Big_data[i].append(round(nb[i]/np.sum(nb),2))# have smthg to record indices where it started
		
	show_species('nb')
	pygame.display.flip()
	clock.tick(30)

plt.show()
