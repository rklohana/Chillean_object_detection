import type {BoundingBox, ImageSize} from './types';

export function clampBox(box: BoundingBox, image: ImageSize): BoundingBox {
  const x1 = Math.max(0, Math.min(image.width, box.x));
  const y1 = Math.max(0, Math.min(image.height, box.y));
  const x2 = Math.max(x1, Math.min(image.width, box.x + box.width));
  const y2 = Math.max(y1, Math.min(image.height, box.y + box.height));
  return {x: x1, y: y1, width: x2 - x1, height: y2 - y1};
}

export function padBox(
  box: BoundingBox,
  image: ImageSize,
  ratio = 0.08,
): BoundingBox {
  const padX = box.width * ratio;
  const padY = box.height * ratio;
  return clampBox(
    {
      x: box.x - padX,
      y: box.y - padY,
      width: box.width + padX * 2,
      height: box.height + padY * 2,
    },
    image,
  );
}

export function intersectionOverUnion(left: BoundingBox, right: BoundingBox): number {
  const x1 = Math.max(left.x, right.x);
  const y1 = Math.max(left.y, right.y);
  const x2 = Math.min(left.x + left.width, right.x + right.width);
  const y2 = Math.min(left.y + left.height, right.y + right.height);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = left.width * left.height + right.width * right.height - intersection;
  return union > 0 ? intersection / union : 0;
}

export type BoxTransform = BoundingBox & {scale: number; offsetX: number; offsetY: number};

export function mapBoxToView(
  box: BoundingBox,
  source: ImageSize,
  view: ImageSize,
  resizeMode: 'contain' | 'cover',
): BoxTransform {
  const scale =
    resizeMode === 'contain'
      ? Math.min(view.width / source.width, view.height / source.height)
      : Math.max(view.width / source.width, view.height / source.height);
  const offsetX = (view.width - source.width * scale) / 2;
  const offsetY = (view.height - source.height * scale) / 2;
  return {
    x: box.x * scale + offsetX,
    y: box.y * scale + offsetY,
    width: box.width * scale,
    height: box.height * scale,
    scale,
    offsetX,
    offsetY,
  };
}
