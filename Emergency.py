f = open("tempX.txt", "r")
g = open("temp2X.txt", "a")
l = f.read().split(".")
l2 = []
i = 0
while i<len(l):
    j = i+(128*128*3)
    l2.append(l[i:j])
    i = j
g.write(str(l2))


f.close()
g.close()
