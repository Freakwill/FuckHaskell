module MyParser
( 

)
where

print_ :: String -> IO String
print_ s = do
  print s
  return s

main :: IO ()
main = do 
   s <- print_ "TH e factorial of 5 is:"
   print s
