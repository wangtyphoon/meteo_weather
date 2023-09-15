diri  = "./"                            ; input directory
fili  = "fanapid.dat"                   ; DSET
fName = diri+fili                       ; path

nlat  = 252                             ; YDEF
mlon  = 252                             ; XDEF
nlev  = 10                              ; LEV
ntim  = 1                               ; REC
vars  = 4                               ; VARS 
UNDEF = 1.e30                           ; UNDEF

; create an array to contain data
x     = new ( (/vars,nlev,nlat,mlon/), float, UNDEF)
setfileoption("bin","ReadByteOrder","BigEndian")
do nt=0,ntim-1                          ; read each record: store in x
   x = fbindirread(fName, nt, (/vars,nlev,nlat,mlon/), "float")
end do

u = x(0,:,:,:)
v = x(1,:,:,:)
w = x(2,:,:,:)
t = x(3,:,:,:)

print(u)