module MySequence
( 

)
where


import Control.Monad


myGet :: String -> IO String
myGet s = do
  putStr s
  getLine

main :: IO ()
main = do 
  x <- myGet "input something:\n"
  replicateM_ 3 (print x)
  replicateM_ 3 (print (x ++ "again"))


