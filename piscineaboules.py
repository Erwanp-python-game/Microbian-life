import pygame
from math import *
import numpy as np
from pygame.locals import *
from random import *
import matplotlib.pyplot as plt

for i in range(1,10):
	a=0
	for j in range(0,i):
		a=a+abs(cos(2*pi*(j+0.5)/i))
	b=0
	for j in range(0,i):
		b=b+abs(cos(2*pi*(j)/i))
	print(max(a,b),i)

xs1=[]
xs2=[]
ys1=[]
ys2=[]
def animate(xs1,ys1,xs2,ys2,I,nourriture):
	xs1.append(I)
	xs2.append(I)
	
	ys1.append(np.mean(nourriture[60*30:60*30+60]))
	ys2.append(np.mean(nourriture[60*34:60*34+60]))
	
	ax1.clear()
	ax2.clear()
	
	ax1.plot(xs1, ys1)
	ax2.plot(xs2, ys2)

L=600
pygame.init()
fenetre = pygame.display.set_mode((L, L))
print(pygame.time.get_ticks())
courantX=np.full((L,L),0.0)
courantY=np.full((L,L),0.0)

courantX0=np.full((3*L,3*L),0.0)
courantY0=np.full((3*L,3*L),0.0)

x,y = np.meshgrid(np.linspace(-30,30,3*L),np.linspace(-30,30,3*L))
speed=3
for i in range(0,5):
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
print(pygame.time.get_ticks())
# courantX=courantX0[L:2*L,L:2*L]
# courantY=courantY0[L:2*L,L:2*L]

subL=60
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
xC=randint(0,L)
yC=randint(0,L)
Vx=0
Vy=0
trace=0
clock = pygame.time.Clock()


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
		
def show_food(actif):
	if actif==1:
		fond2=fond.copy()
		listim=[]
		im=pygame.Surface((L//subL,L//subL),pygame.SRCALPHA, 32)
		for i in range(0,subL**2):
			im.fill((0,min(max(int(200*nourriture[i]),0),255),200,100))
			
			listim.append((im.copy(),((L//subL)*(i//subL),(L//subL)*(i%subL))))
		
		fond2.blits(listim)
		return fond2
	else:
		return fond

def grad(I):
	return (nourriture[(I+1)%3600]-nourriture[(I-1)%3600],nourriture[(I+60)%3600]-nourriture[(I-60)%3600])

class Organism():
	def __init__(self):
		self.xc=randint(0,L-1)
		self.yc=randint(0,L-1)
		self.color=(randint(0,255),randint(0,255),randint(0,255))
		self.vx=0
		self.vy=0
		
	def move_eat(self):
		I1=int(self.yc)//10+60*(int(self.xc)//10)
		G=grad(I1)
		self.vx=(0.5*self.vx+0.5*courantY[int(self.xc)][int(self.yc)])/1.2+1*np.sign(G[1])
		self.vy=(0.5*self.vy+0.5*courantX[int(self.xc)][int(self.yc)])/1.2+1*np.sign(G[0])
		self.xc=(self.xc+self.vx)%L
		self.yc=(self.yc+self.vy)%L
		nourriture[I1]=max((nourriture[I1]-10),0)
	
	def draw(self,trace):
		if trace==1:
			pygame.draw.circle(fond,self.color,(int(self.xc),int(self.yc)),2)
		pygame.draw.circle(fenetre,self.color,(int(self.xc),int(self.yc)),4)

All_Org=[]
for i in range(0,300):
	All_Org.append(Organism())

I=-1
fenetre.blit(fond,(0,0))
while q==0:
	T1=pygame.time.get_ticks()
	clock.tick(30)
	I+=1
	#print(I%100)
	# if (I%100)==0:
		# print('a')
		# fig = plt.figure()
		# ax1 = fig.add_subplot(1, 2, 1)
		# ax2 = fig.add_subplot(1, 2, 2)
		# animate(xs1,ys1,xs2,ys2,I,nourriture)
		# plt.show()
	
	#pygame.time.wait(100)
	xC=xC%L
	yC=yC%L
	nourriture=np.dot(flux,nourriture)
	nourriture=np.absolute(nourriture)*(30000/np.absolute(nourriture).sum())

	for event in pygame.event.get():
		if event.type == QUIT:
			q=1
			plt.hist(nourriture,bins=100,range=(0,20))
			plt.show()
	fenetre.fill((0,20,150))
	Im=show_food(1)
	fenetre.blit(Im,(0,0))
	
	for i in All_Org:
		i.move_eat()
		i.draw(trace)
	
	# if trace==1:
		# pygame.draw.circle(fond,(255,0,0),(int(xC),int(yC)),5)
	# pygame.draw.circle(fenetre,(255,0,0),(int(xC),int(yC)),5)
	print(pygame.time.get_ticks()-T1)
	pygame.display.flip()
