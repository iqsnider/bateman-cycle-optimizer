#import "@preview/slydst:0.1.4": *
#show: slides.with(
  title: [BANG Meeting],
  authors: ("Ian Snider",), layout: "large",
  ratio: 4/3,
  title-color: none,
)

#set page(margin : (x : 0.5cm, y : 0.5cm))

= BANG Updates \ 11.07.2025

==  Bateman
#v(5em)
Ba beam -> implanted on tape in detector zone -> observed coincidences -> fresh piece of tape cycled in after $t_"cycle"$ to prevent contamination
$ dot(N)_p = R - lambda N_p \ dot(N)_d = lambda_p N_p - lambda_d N_d $

$N_p (0) = N_d (0) = 0 $


`scipy.integrate.solve_ivp`

$""^(147)_()"Ba"$ $t_(1/2) = 0.894$ s #h(2em) $""^(147)_()"La"$ $t_(1/2) = 4.06$ s  \
$""^(148)_()"Ba"$ $t_(1/2) = 0.619$ s #h(2em) $""^(148)_()"La"$ $t_(1/2) = 1.411$ s  \

maximize $"SNR"(t_"cycle") = frac(integral_0^t_"cycle" A_p dif t ,integral_0^t_"cycle" A_d dif t )$ might have to specify minimum contamination

Find maximum net $A_p$

// because I love LTI models:
//
// $ mat(dot(N)_p; dot(N)_d) = mat(-lambda_p, 0; 0 ,-lambda_d)mat(N_p; N_d) + mat(1;0)R \
// mat(A_p; A_d) = mat(lambda_p, 0; 0, lambda_d)mat(N_p;N_d) $

#image("figs/A147.png")
#image("figs/A148.png")
#image("figs/A147_max.png")
#image("figs/A148_max.png")

== Notes
#v(5em)
simulate detector response with monte carlo
- roll dice -> did it decay
  - did we observe it?
- optimize $t_"cycle"$ over the entire experiment duration
  - observations will be periodic
- refer to adriannas literature for taping cycle times
- report total number of counts from the parent and daughter for various $t_"cycle"$ s
  - then find the maximum

= BANG Updates \ 11.14.2025

== Tape timings
#v(5em)
Previous experiments had daughter nuclei with minutes half-lives

$""^92$Sr$(n,gamma)^93$Sr contamination level:
$ C = frac(integral_0^t_"cycle" A_d (t)  dif t ,integral_0^t_"cycle" A_p (t) + A_d (t) dif t)  = 3.85% $ 
$t_"cycle" = 60$ s 

Found daughter $beta-gamma$ coincidence matrix and subtracted from the parent matrix

Will make monte carlo simulation


What is her plan with 148La, is she aware of the low statistics?
How is she defining good statistics?
Conclusion for presentation, will be next step.
- simulations
- raineer
- geant4

What don't you just cross a neutron beam with a La ion beam? Is it too expensive to maintain the neutron beam?

$ ""^(147)_()"Ba" arrow ""^(147)_()"La" + beta^(-) + overline(nu_"e") $ 
