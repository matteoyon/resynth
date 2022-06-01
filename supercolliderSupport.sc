s.boot

m = 2

(
a = {SinOsc.ar(416.628,0,0.100024/m)};
b = {SinOsc.ar(416.872,0,0.458049/m)};
c = {SinOsc.ar(420.642,0,1/m)};
d = {SinOsc.ar(1169.3,0,0.292743/m)};
e = {SinOsc.ar(2174.61,0,0.147832/m)};
)

(
a.play;
b.play;
c.play;
d.play;
e.play;
)


