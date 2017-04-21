# installing/loading the package:
if(!require(installr))
{
  #load / install+load installr:
  install.packages("installr");
  require(installr)
}
# install, move, update.package, quit R:
require(installr)
updateR() # updateR(F, T, T, F, T, F, T)
