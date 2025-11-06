#import "@preview/slydst:0.1.4": *
#show: slides.with(
  title: [BANG Meeting],
  authors: ("Ian Snider",),
  layout: "large",
  ratio: 4/3,
  title-color: none,
)

#set page(margin : (x : 0.5cm, y : 0.5cm))

= BANG Updates \ 10.31.2025

==  Bateman
#v(5em)
$ dot(N)_p = R - lambda N_p \ dot(N)_d = lambda_p N_p - lambda_d N_d $

$N_p (0) = N_d (0) = 0 $


`scipy.integrate.solve_ivp`

$""^(147)_()"Ba"$ $t_(1/2) = 0.894$ s #h(2em) $""^(147)_()"La"$ $t_(1/2) = 4.06$ s  \
$""^(148)_()"Ba"$ $t_(1/2) = 0.619$ s #h(2em) $""^(148)_()"La"$ $t_(1/2) = 1.411$ s  \

maximize $"SNR"(t_"cycle") = frac(integral_0^t_"cycle" A_p dif t ,integral_0^t_"cycle" A_d dif t ) $ might have to specify minimum contamination
// because I love LTI models:
//
// $ mat(dot(N)_p; dot(N)_d) = mat(-lambda_p, 0; 0 ,-lambda_d)mat(N_p; N_d) + mat(1;0)R \
// mat(A_p; A_d) = mat(lambda_p, 0; 0, lambda_d)mat(N_p;N_d) $
