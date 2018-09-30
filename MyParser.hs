module MyParser
( 
   sSplitHead,
   fCheck,
   sCheck,
   sSkip,
   spaceSkip
)
where

import qualified Data.Strings as S
import Control.Monad.State.Lazy

type ParseResult = String | [ParseResult]

type Parser = State String (Maybe ParseResult)

instance Monoid Parser where
  <> parser1 parser2 = do pr <- parser1
    (\p -> (pr ++ p)) <$> parser2

sSplitHead :: String -> Parser
sSplitHead s = state $ \t -> if S.sStartsWith t s then (s, drop (length s) t) else (Nothing, "")

fCheck0 :: (Char -> Bool) -> Parser
fCheck0 f = state $ \s -> if (s == "") then (Nothing, "") else (let s=c:t in if (f c) then (Just [c], t) else (Nothing, s))

fCheck :: (Char -> Bool) -> Parser

fCheck f = (fCheck0 f) <> (fCheck0 f)

sCheck :: String -> Parser
sCheck t = fCheck (`elem` t)

sSkip :: String -> String -> String
sSkip t = dropWhile (`elem` t)

spaceSkip :: String -> String
spaceSkip = sSkip " \t\n"

skipSpacewhits = modify spaceSkip


par :: Parser
par = do
    pr <- sSplitHead "T"
    (\p -> (pr ++ p)) <$> (sSplitHead "H")

main :: IO ()
main = do 
   print (runState par " TH e factorial of 5 is:")
