module Core.Stride
  ( isContiguousLayout
  , defaultStride
  , transpose2
  , permute
  , reshape
  , slice
  , select
  , expand
  ) where

import Error
import Core.Tensor (Tensor)

isContiguousLayout :: a -> Bool
isContiguousLayout _ = False

defaultStride :: [Int] -> [Int]
defaultStride _ = error "NYI: defaultStride"

transpose2 :: Tensor a -> Int -> Int -> Either TensorError (Tensor a)
transpose2 _ _ _ = Left (NotImplemented "transpose2")

permute :: Tensor a -> [Int] -> Either TensorError (Tensor a)
permute _ _ = Left (NotImplemented "permute")

reshape :: Tensor a -> [Int] -> Either TensorError (Tensor a)
reshape _ _ = Left (NotImplemented "reshape")

slice :: Tensor a -> [(Int, Int, Int)] -> Either TensorError (Tensor a)
slice _ _ = Left (NotImplemented "slice")

select :: Tensor a -> Int -> Int -> Either TensorError (Tensor a)
select _ _ _ = Left (NotImplemented "select")

expand :: Tensor a -> [Int] -> Either TensorError (Tensor a)
expand _ _ = Left (NotImplemented "expand")
