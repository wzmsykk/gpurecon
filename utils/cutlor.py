











endcount=160000
fp=open("singles_ascii.dat.cid","r")
lines=fp.readlines()
fpw=open("cut.lor","w")
fpw.writelines(lines[:endcount])
fp.close()
fpw.close()